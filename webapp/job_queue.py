"""In-process conversion queue, per-IP budget and job retention.

A conversion takes far longer than an HTTP response may live behind
Cloudflare, so ``/api/convert`` only accepts the take.  One worker thread
drains the queue and writes ``out/status.json`` as the job advances; the
browser polls that file through ``/api/jobs/<id>/status.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Callable

# Job states.  ``done`` and ``error`` are terminal.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"

RETENTION_DAYS = int(os.environ.get("TABULA_JOB_RETENTION_DAYS", "7"))
MAX_JOBS_GB = float(os.environ.get("TABULA_JOBS_MAX_GB", "5"))
RATE_PER_HOUR = int(os.environ.get("TABULA_RATE_PER_HOUR", "6"))
RATE_PER_DAY = int(os.environ.get("TABULA_RATE_PER_DAY", "20"))

STATUS_NAME = "status.json"


class QueueFull(Exception):
    """The waiting queue has no room for another take."""


class JobFailed(Exception):
    """A conversion failure whose message is safe to show in the UI."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


def status_path(job_dir: Path) -> Path:
    return Path(job_dir) / "out" / STATUS_NAME


def write_status(job_dir: Path, payload: dict) -> None:
    """Replace the job status atomically so a poll never reads half a file."""
    path = status_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_status(job_dir: Path) -> dict | None:
    try:
        payload = json.loads(status_path(job_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class _Job:
    def __init__(self, job_id: str, job_dir: Path, run: Callable[[], dict], message: str):
        self.id = job_id
        self.job_dir = Path(job_dir)
        self.run = run
        self.message = message
        self.state = QUEUED
        self.result: dict | None = None
        self.error: str | None = None
        self.detail: str | None = None

    def payload(self, position: int | None) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "queue_position": position,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "detail": self.detail,
        }


class JobQueue:
    """One worker, at most ``max_waiting`` takes waiting behind it."""

    def __init__(self, max_waiting: int = 2):
        self.max_waiting = max_waiting
        self._queue: Queue = Queue()
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._waiting: list[_Job] = []
        self._running: _Job | None = None
        self._thread: threading.Thread | None = None

    # -- public ---------------------------------------------------------
    def submit(
        self,
        job_id: str,
        job_dir: Path,
        run: Callable[[], dict],
        message: str = "Waiting for the tab worker…",
    ) -> dict:
        """Queue a take and return its status payload, or raise ``QueueFull``."""
        job = _Job(job_id, job_dir, run, message)
        with self._lock:
            # A burst can arrive before the worker claims the first take: the
            # job at the head of the queue will be the running one, so it does
            # not use a waiting slot.
            limit = self.max_waiting + (0 if self._running is not None else 1)
            if len(self._waiting) >= limit:
                raise QueueFull()
            self._jobs[job_id] = job
            self._waiting.append(job)
            self._publish_locked(job)
            self._queue.put(job)
            self._ensure_worker_locked()
            return job.payload(self._position_locked(job))

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.payload(self._position_locked(job))

    def active(self) -> set[str]:
        with self._lock:
            return {self._running.id} if self._running is not None else set()

    # -- internals ------------------------------------------------------
    def _ensure_worker_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._worker, name="tabula-convert", daemon=True
        )
        self._thread.start()

    def _position_locked(self, job: _Job) -> int | None:
        """0-based place in the pending order, or None once the job is over."""
        if job.state == RUNNING:
            return 0
        if job.state != QUEUED or job not in self._waiting:
            return None
        ahead = self._waiting.index(job)
        return ahead + (1 if self._running is not None else 0)

    def _publish_locked(self, job: _Job) -> None:
        try:
            write_status(job.job_dir, job.payload(self._position_locked(job)))
        except OSError as exc:  # polling still works from memory
            print(f"jobs: could not write status for {job.id}: {exc}", flush=True)

    def _finish(self, job: _Job, state: str, **fields) -> None:
        with self._lock:
            job.state = state
            for key, value in fields.items():
                setattr(job, key, value)
            self._running = None
            self._publish_locked(job)
            for waiting in self._waiting:
                self._publish_locked(waiting)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                with self._lock:
                    if job in self._waiting:
                        self._waiting.remove(job)
                    self._running = job
                    job.state = RUNNING
                    job.message = "Building the tab…"
                    self._publish_locked(job)
                    for waiting in self._waiting:
                        self._publish_locked(waiting)
                try:
                    result = job.run()
                except JobFailed as exc:
                    self._finish(
                        job, ERROR,
                        message=exc.message,
                        error=exc.message,
                        detail=exc.detail,
                    )
                except BaseException:  # noqa: BLE001 - a dead worker would hang the queue
                    tail = traceback.format_exc().strip()[-4000:]
                    self._finish(
                        job, ERROR,
                        message="The conversion crashed.",
                        error="The conversion crashed.",
                        detail=tail,
                    )
                else:
                    self._finish(
                        job, DONE,
                        message="Done.",
                        result=result,
                        error=None,
                        detail=None,
                    )
            finally:
                self._queue.task_done()


class Budget:
    """Per-IP conversion budget kept in SQLite.

    Cloudflare's free plan has no WAF rule to lean on, so the limit lives
    here.  Counts are taken when a take is accepted, never when rejected.
    """

    def __init__(
        self,
        path: Path,
        per_hour: int = RATE_PER_HOUR,
        per_day: int = RATE_PER_DAY,
    ):
        self.path = Path(path)
        self.per_hour = per_hour
        self.per_day = per_day
        self._lock = threading.Lock()
        self._last_clean = 0.0

    def blocked(self, ip: str) -> str | None:
        """Return a user-facing reason when this address is out of budget."""
        if self.per_hour <= 0 and self.per_day <= 0:
            return None
        with self._lock:
            now = int(time.time())
            try:
                conn = self._connect()
            except sqlite3.Error as exc:
                # A broken budget file must not stop the app converting.
                print(f"budget: unavailable ({exc}); allowing the take", flush=True)
                return None
            try:
                self._maybe_clean(conn, now)
                for window, limit in (("hour", self.per_hour), ("day", self.per_day)):
                    if limit <= 0:
                        continue
                    used = self._used(conn, ip, window, self._bucket(now, window))
                    if used >= limit:
                        unit = "conversion" if limit == 1 else "conversions"
                        return (
                            f"That address has used its {limit} {unit} "
                            f"this {window}. Try again later."
                        )
                return None
            finally:
                conn.close()

    def consume(self, ip: str) -> None:
        if self.per_hour <= 0 and self.per_day <= 0:
            return
        with self._lock:
            now = int(time.time())
            try:
                conn = self._connect()
            except sqlite3.Error as exc:
                print(f"budget: unavailable ({exc}); take not counted", flush=True)
                return
            try:
                for window in ("hour", "day"):
                    conn.execute(
                        "INSERT INTO budget (ip, window, window_start, used)"
                        " VALUES (?, ?, ?, 1)"
                        " ON CONFLICT(ip, window, window_start)"
                        " DO UPDATE SET used = used + 1",
                        (ip, window, self._bucket(now, window)),
                    )
                conn.commit()
            except sqlite3.Error as exc:
                print(f"budget: could not record the take ({exc})", flush=True)
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS budget ("
            " ip TEXT NOT NULL,"
            " window TEXT NOT NULL,"
            " window_start INTEGER NOT NULL,"
            " used INTEGER NOT NULL,"
            " PRIMARY KEY (ip, window, window_start))"
        )
        return conn

    @staticmethod
    def _bucket(now: int, window: str) -> int:
        size = 3600 if window == "hour" else 86400
        return now // size * size

    @staticmethod
    def _used(conn: sqlite3.Connection, ip: str, window: str, start: int) -> int:
        row = conn.execute(
            "SELECT used FROM budget"
            " WHERE ip = ? AND window = ? AND window_start = ?",
            (ip, window, start),
        ).fetchone()
        return int(row[0]) if row else 0

    def _maybe_clean(self, conn: sqlite3.Connection, now: int) -> None:
        if now - self._last_clean < 3600:
            return
        self._last_clean = float(now)
        conn.execute("DELETE FROM budget WHERE window_start < ?", (now - 2 * 86400,))
        conn.commit()


def recover_jobs(jobs_dir: Path) -> int:
    """Fail jobs left mid-flight by a previous process so polls never hang."""
    root = Path(jobs_dir)
    if not root.is_dir():
        return 0
    count = 0
    for job_dir in root.iterdir():
        if not job_dir.is_dir():
            continue
        status = read_status(job_dir)
        if status is None or status.get("state") not in {QUEUED, RUNNING}:
            continue
        status.update(
            {
                "state": ERROR,
                "queue_position": None,
                "message": "The server restarted before this take finished.",
                "result": None,
                "error": "The server restarted.",
                "detail": None,
            }
        )
        try:
            write_status(job_dir, status)
        except OSError:
            continue
        count += 1
    return count


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            pass
    return total


def cleanup_jobs(
    jobs_dir: Path,
    running: set[str] | frozenset[str] = frozenset(),
    retention_days: int = RETENTION_DAYS,
    max_gb: float = MAX_JOBS_GB,
) -> dict:
    """Drop old takes, then the oldest ones until the total fits the cap."""
    root = Path(jobs_dir)
    if not root.is_dir():
        return {"removed": 0, "freed": 0, "kept": 0}
    now = time.time()
    cutoff = now - retention_days * 86400
    entries: list[tuple[Path, float, int]] = []
    for path in root.iterdir():
        try:
            if not path.is_dir():
                continue
            entries.append((path, path.stat().st_mtime, _dir_size(path)))
        except OSError:
            continue

    removed = 0
    freed = 0
    kept: list[tuple[Path, float, int]] = []
    for path, mtime, size in entries:
        if path.name in running:
            kept.append((path, mtime, size))
            continue
        if retention_days > 0 and mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
            freed += size
        else:
            kept.append((path, mtime, size))

    total = sum(size for _, _, size in kept)
    cap = int(max_gb * 1024**3)
    if cap > 0 and total > cap:
        left: list[tuple[Path, float, int]] = []
        for path, mtime, size in sorted(kept, key=lambda item: item[1]):
            if total <= cap or path.name in running:
                left.append((path, mtime, size))
                continue
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            removed += 1
            freed += size
        kept = left

    if removed:
        print(
            f"Jobs: removed {removed} take(s), freed {freed / 1024**2:.1f} MB, "
            f"{len(kept)} kept.",
            flush=True,
        )
    else:
        print(f"Jobs: nothing to remove, {len(kept)} take(s) kept.", flush=True)
    return {"removed": removed, "freed": freed, "kept": len(kept)}
