"""Minimal local web app: MIDI upload -> Guitar Pro tab -> alphaTab playback."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

try:
    from webapp.gp5 import build_gp5
except ImportError:  # Allow ``python webapp/server.py`` as well as ``-m webapp.server``.
    from gp5 import build_gp5

REPO = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
JOBS = WEB / "jobs"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO / path)


CHECKPOINT = _project_path(os.environ.get(
    "TABULA_CHECKPOINT", "runs/finetune_t5_guitarset/best"
))
VOCAB = _project_path(os.environ.get("TABULA_VOCAB", "vocab/canonical.json"))
PYTHON = os.environ.get("TABULA_PYTHON", sys.executable)
MAX_UPLOAD_MB = int(os.environ.get("TABULA_MAX_UPLOAD_MB", "32"))
CONVERT_TIMEOUT = int(os.environ.get("TABULA_CONVERT_TIMEOUT", "900"))

CONVERT_LOCK = threading.Lock()
JOBS.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(WEB / "static"),
    static_url_path="/static",
    template_folder=str(WEB / "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Python's mimetypes table is missing a few types used by the vendored player.
for suffix, mime in (
    (".mjs", "text/javascript"),
    (".sf2", "audio/x-soundfont"),
    (".woff2", "font/woff2"),
    (".woff", "font/woff"),
    (".otf", "font/otf"),
    (".gp5", "application/x-guitar-pro"),
    (".mid", "audio/midi"),
    (".midi", "audio/midi"),
):
    mimetypes.add_type(mime, suffix)


def _cleanup_jobs(max_age_seconds: int = 24 * 60 * 60) -> None:
    now = time.time()
    for path in JOBS.iterdir():
        try:
            if path.is_dir() and now - path.stat().st_mtime > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _summary(report: dict, records: list[dict], exported: dict) -> dict:
    capo = int(report.get("capo", 0))
    trials = report.get("capo_scores") or []
    selected = next((item for item in trials if item.get("capo") == capo), {})
    open_notes = selected.get("open_notes")
    if open_notes is None:
        open_notes = sum(1 for note in records if int(note.get("f", -1)) == 0)
    notes = int(report.get("final_notes") or len(records))
    return {
        "capo": capo,
        "feasible_max": report.get("capo_feasible_max"),
        "notes": notes,
        "open_notes": open_notes,
        "open_rate": selected.get("open_rate"),
        "high_notes": selected.get("high_notes"),
        "high_rate": selected.get("high_rate"),
        "max_fret": selected.get("max_fret"),
        "mean_fret": selected.get("mean_fret"),
        "dropped": len(report.get("dropped_chord_tones") or []),
        "postprocessing": report.get("postprocessing") or {},
        "raw_difficulty": report.get("raw_difficulty"),
        "exported": exported,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        max_upload_mb=MAX_UPLOAD_MB,
        checkpoint_ready=CHECKPOINT.exists(),
        vocab_ready=VOCAB.exists(),
    )


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_ready": CHECKPOINT.exists(),
        "vocab": str(VOCAB),
        "vocab_ready": VOCAB.exists(),
        "jobs": len([p for p in JOBS.iterdir() if p.is_dir()]),
    })


def _error(message: str, status: int = 400, detail: str | None = None):
    payload = {"error": message}
    if detail:
        payload["detail"] = detail
    return jsonify(payload), status


@app.post("/api/convert")
def convert():
    if not CHECKPOINT.exists():
        return _error(
            "The tab model checkpoint is missing.",
            500,
            f"Expected: {CHECKPOINT}",
        )
    if not VOCAB.exists():
        return _error(
            "The tab model vocabulary is missing.",
            500,
            f"Expected: {VOCAB}",
        )

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _error("Choose a MIDI file first.")

    original_name = Path(upload.filename).name
    safe_name = secure_filename(original_name) or "upload.mid"
    if Path(safe_name).suffix.lower() not in {".mid", ".midi"}:
        return _error("Upload a .mid or .midi file.")

    job_id = uuid.uuid4().hex
    job_dir = JOBS / job_id
    out_dir = job_dir / "out"
    job_dir.mkdir(parents=True, exist_ok=False)
    upload_path = job_dir / "upload.mid"
    try:
        upload.save(upload_path)
    except OSError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error("The upload could not be saved.", 500, str(exc))
    if upload_path.stat().st_size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error("That MIDI file is empty.")

    command = [
        PYTHON, str(REPO / "scripts" / "midi2tab.py"), str(upload_path),
        "--checkpoint", str(CHECKPOINT),
        "--vocab", str(VOCAB),
        "--out-dir", str(out_dir),
        "--beam", "4",
        "--max-fret", "6",
        "--capo", "auto",
        "--max-capo", "6",
        "--width", "120",
        "--no-audio",
    ]
    env = os.environ.copy()
    src = str(REPO / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = time.time()
    try:
        with CONVERT_LOCK:
            result = subprocess.run(
                command,
                cwd=str(REPO),
                env=env,
                text=True,
                capture_output=True,
                timeout=CONVERT_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        return _error(
            "The conversion timed out.",
            504,
            f"The model did not finish within {CONVERT_TIMEOUT} seconds.",
        )
    except OSError as exc:
        return _error("The conversion process could not start.", 500, str(exc))

    if result.returncode != 0:
        log = (result.stderr or "") + ("\n" + result.stdout if result.stdout else "")
        return _error(
            "The tab model could not process this MIDI.",
            500,
            log[-7000:].strip() or f"Exit code {result.returncode}",
        )

    final_json = out_dir / "final.json"
    report_json = out_dir / "report.json"
    final_mid = out_dir / "final.mid"
    if not final_json.exists() or not report_json.exists() or not final_mid.exists():
        return _error(
            "The conversion finished without its expected artifacts.",
            500,
            f"Missing one of final.json, report.json, final.mid in {out_dir}",
        )

    try:
        records = json.loads(final_json.read_text(encoding="utf-8"))
        report = json.loads(report_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _error("The conversion artifacts could not be read.", 500, str(exc))

    title = Path(original_name).stem or "Uploaded MIDI"
    capo = int(report.get("capo", 0))
    try:
        exported = build_gp5(
            records,
            final_mid,
            job_dir / "score.gp5",
            capo=capo,
            title=title,
        )
    except Exception as exc:  # noqa: BLE001 - surface a useful local error
        return _error("The Guitar Pro tab could not be written.", 500, str(exc))

    elapsed = round(time.time() - started, 1)
    base = f"/api/jobs/{job_id}"
    return jsonify({
        "id": job_id,
        "title": title,
        "capo": capo,
        "elapsed": elapsed,
        "summary": _summary(report, records, exported),
        "urls": {
            "gp5": f"{base}/score.gp5",
            "midi": f"{base}/final.mid",
            "tab": f"{base}/final_tab.txt",
            "report": f"{base}/report.json",
        },
    })


_ALLOWED_FILES = {
    "score.gp5": ("score.gp5", "application/x-guitar-pro"),
    "final.mid": ("out/final.mid", "audio/midi"),
    "final_tab.txt": ("out/final_tab.txt", "text/plain; charset=utf-8"),
    "report.json": ("out/report.json", "application/json"),
    "upload.mid": ("upload.mid", "audio/midi"),
}


@app.get("/api/jobs/<job_id>/<name>")
def job_file(job_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    allowed = _ALLOWED_FILES.get(name)
    if allowed is None:
        abort(404)
    relative, mime = allowed
    path = JOBS / job_id / relative
    if not path.is_file():
        abort(404)
    return send_file(
        path,
        mimetype=mime,
        as_attachment=name != "score.gp5",
        download_name=name,
        conditional=True,
    )


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({
        "error": f"That file is larger than the {MAX_UPLOAD_MB} MB limit.",
    }), 413


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found."}), 404
    return "Not found", 404


def main() -> int:
    parser = argparse.ArgumentParser(description="Local MIDI-to-tab web app")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _cleanup_jobs()
    shown_host = args.host
    if shown_host in {"0.0.0.0", "::"}:
        shown_host = socket.gethostname()
    print(f"Tabula web app: http://{shown_host}:{args.port}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Vocab:      {VOCAB}")
    if not CHECKPOINT.exists() or not VOCAB.exists():
        print("WARNING: checkpoint or vocabulary is missing; conversions will fail.")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
