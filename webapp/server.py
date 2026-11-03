"""Local web app: source audio + MIDI transcription -> Guitar Pro tab."""

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
    from webapp.gp5 import build_gp5 as build_legacy_gp5
except ImportError:  # Allow ``python webapp/server.py`` as well as ``-m webapp.server``.
    from gp5 import build_gp5 as build_legacy_gp5

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
MAX_AUDIO_MB = int(os.environ.get("TABULA_MAX_AUDIO_MB", "96"))
MAX_MIDI_MB = int(os.environ.get("TABULA_MAX_MIDI_MB", "8"))
MAX_UPLOAD_MB = int(
    os.environ.get("TABULA_MAX_UPLOAD_MB", str(MAX_AUDIO_MB + MAX_MIDI_MB + 16))
)
CONVERT_TIMEOUT = int(os.environ.get("TABULA_CONVERT_TIMEOUT", "900"))
AUDIO_TIMEOUT = int(os.environ.get("TABULA_AUDIO_TIMEOUT", "180"))

AUDIO_SUFFIXES = {
    ".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma",
}
MIDI_SUFFIXES = {".mid", ".midi"}

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
    (".wav", "audio/wav"),
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


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_notes(path: Path) -> list[dict]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        payload = payload.get("notes", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a note list")
    return payload


def _summary(
    report: dict,
    records: list[dict],
    exported: dict,
    pipeline: dict | None = None,
) -> dict:
    capo = int(report.get("capo", 0))
    trials = report.get("capo_scores") or []
    selected = next((item for item in trials if item.get("capo") == capo), {})
    open_notes = selected.get("open_notes")
    if open_notes is None:
        open_notes = sum(1 for note in records if int(note.get("f", -1)) == 0)
    notes = int(report.get("final_notes") or len(records))
    pipeline = pipeline or {}
    alignment = pipeline.get("alignment") or {}
    dead_notes = pipeline.get("dead_notes", exported.get("dead_notes", 0))
    hp = pipeline.get("hp_applied", 0)
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
        "dead_notes": int(dead_notes or 0),
        "hp": int(hp or 0),
        "match_rate": alignment.get("match_rate"),
        "offset_sec": alignment.get("applied_offset_sec"),
        "grid_ticks": pipeline.get("grid_ticks"),
        "exported": exported,
    }


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _normalize_audio(source: Path, destination: Path) -> Path:
    """Decode any browser-friendly audio file to a small mono WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # librosa can read WAV/FLAC/OGG natively. If ffmpeg is missing, let the
        # pipeline try the original file rather than rejecting a valid take.
        if source.suffix.lower() in {".wav", ".flac", ".ogg", ".oga"}:
            return source
        raise RuntimeError("ffmpeg is required to decode this audio format.")

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "22050",
        "-c:a", "pcm_s16le", str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=AUDIO_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Audio decoding took longer than {AUDIO_TIMEOUT} seconds."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start ffmpeg: {exc}") from exc
    if result.returncode != 0 or not destination.exists():
        log = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(log[-3000:] or "ffmpeg could not decode the recording.")
    return destination


def _run_conversion(command: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    src = str(REPO / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with CONVERT_LOCK:
        return subprocess.run(
            command,
            cwd=str(REPO),
            env=env,
            text=True,
            capture_output=True,
            timeout=CONVERT_TIMEOUT,
        )


@app.get("/")
def index():
    return render_template(
        "index.html",
        max_audio_mb=MAX_AUDIO_MB,
        max_midi_mb=MAX_MIDI_MB,
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
        "max_audio_mb": MAX_AUDIO_MB,
        "max_midi_mb": MAX_MIDI_MB,
    })


def _error(message: str, status: int = 400, detail: str | None = None):
    payload = {"error": message}
    if detail:
        payload["detail"] = detail
    return jsonify(payload), status


def _conversion_error_message(log: str, audio_mode: bool) -> str:
    lowered = log.lower()
    if "match rate" in lowered:
        return "The recording and MIDI do not line up."
    if audio_mode and ("audio" in lowered or "ffmpeg" in lowered):
        return "The recording could not be read."
    return "The tab could not be built from this take."


def _validate_upload(
    upload,
    *,
    label: str,
    suffixes: set[str],
    max_mb: int,
) -> tuple[Path, str] | None:
    name = Path(upload.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in suffixes:
        if ".mid" in suffixes:
            raise ValueError(f"Choose a MIDI (.mid or .midi) file for the {label}.")
        raise ValueError(f"Choose an audio file for the {label}.")
    # FileStorage streams are rewindable; reject before writing when the
    # browser supplied a length.
    length = getattr(upload, "content_length", None)
    if length and length > max_mb * 1024 * 1024:
        raise ValueError(f"The {label} is larger than {max_mb} MB.")
    safe = secure_filename(name) or f"{label}{suffix}"
    return Path(safe), suffix


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

    midi_upload = request.files.get("midi") or request.files.get("file")
    audio_upload = request.files.get("audio")

    if midi_upload is None or not midi_upload.filename:
        return _error("Choose the MIDI transcription for this take.")
    try:
        midi_original, midi_suffix = _validate_upload(
            midi_upload,
            label="MIDI transcription",
            suffixes=MIDI_SUFFIXES,
            max_mb=MAX_MIDI_MB,
        )
    except ValueError as exc:
        return _error(str(exc))

    audio_mode = bool(audio_upload and audio_upload.filename)
    audio_original = None
    audio_suffix = ""
    if audio_mode:
        try:
            audio_original, audio_suffix = _validate_upload(
                audio_upload,
                label="source recording",
                suffixes=AUDIO_SUFFIXES,
                max_mb=MAX_AUDIO_MB,
            )
        except ValueError as exc:
            return _error(str(exc))

    job_id = uuid.uuid4().hex
    job_dir = JOBS / job_id
    out_dir = job_dir / "out"
    job_dir.mkdir(parents=True, exist_ok=False)

    midi_name = midi_original.name if midi_original else f"transcription{midi_suffix}"
    midi_path = job_dir / f"transcribed{midi_suffix}"
    audio_path = None
    normalized_audio = None
    if audio_mode:
        audio_name = audio_original.name if audio_original else f"recording{audio_suffix}"
        audio_path = job_dir / f"upload_audio{audio_suffix}"
        normalized_audio = job_dir / "source.wav"
    else:
        audio_name = None

    try:
        midi_upload.save(midi_path)
        if midi_path.stat().st_size == 0:
            raise ValueError("That MIDI file is empty.")
        if _size_mb(midi_path) > MAX_MIDI_MB:
            raise ValueError(f"The MIDI transcription is larger than {MAX_MIDI_MB} MB.")
        if audio_mode:
            audio_upload.save(audio_path)
            if audio_path.stat().st_size == 0:
                raise ValueError("That recording is empty.")
            if _size_mb(audio_path) > MAX_AUDIO_MB:
                raise ValueError(f"The source recording is larger than {MAX_AUDIO_MB} MB.")
            normalized_audio = _normalize_audio(audio_path, normalized_audio)
    except (OSError, ValueError, RuntimeError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error(str(exc) or "The upload could not be saved.", 400)
    except Exception as exc:  # noqa: BLE001 - surfaced as a local error
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error("The upload could not be saved.", 500, str(exc))

    title = Path(audio_name or midi_name).stem or "Untitled take"
    started = time.time()

    if audio_mode:
        command = [
            PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
            "--audio", str(normalized_audio),
            "--midi", str(midi_path),
            "--out-dir", str(out_dir),
            "--stage", "all",
            "--checkpoint", str(CHECKPOINT),
            "--vocab", str(VOCAB),
            "--title", title,
            "--beam", "4",
            "--width", "120",
            "--force",
        ]
    else:
        command = [
            PYTHON, str(REPO / "scripts" / "midi2tab.py"), str(midi_path),
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

    try:
        result = _run_conversion(command)
    except subprocess.TimeoutExpired:
        return _error(
            "The conversion timed out.",
            504,
            f"The take did not finish within {CONVERT_TIMEOUT} seconds.",
        )
    except OSError as exc:
        return _error("The conversion process could not start.", 500, str(exc))

    if result.returncode != 0:
        log = (result.stderr or "") + ("\n" + result.stdout if result.stdout else "")
        log = log[-8000:].strip() or f"Exit code {result.returncode}"
        return _error(_conversion_error_message(log, audio_mode), 500, log)

    pipeline = None
    if audio_mode:
        final_json = out_dir / "tabula" / "final.json"
        report_json = out_dir / "tabula" / "report.json"
        final_mid = out_dir / "tabula" / "final.mid"
        gp5_path = out_dir / "song.gp5"
        tab_path = out_dir / "tab_with_features.txt"
        pipeline_path = out_dir / "pipeline_report.json"
        required = (final_json, report_json, final_mid, gp5_path, tab_path, pipeline_path)
        missing = [str(path.name) for path in required if not path.exists()]
        if missing:
            return _error(
                "The conversion finished without its expected artifacts.",
                500,
                f"Missing: {', '.join(missing)} in {out_dir}",
            )
        try:
            pipeline = _load_json(pipeline_path)
            exported = pipeline.get("gp5") or {}
        except (OSError, ValueError) as exc:
            return _error("The pipeline report could not be read.", 500, str(exc))
    else:
        final_json = out_dir / "final.json"
        report_json = out_dir / "report.json"
        final_mid = out_dir / "final.mid"
        tab_path = out_dir / "final_tab.txt"
        gp5_path = job_dir / "score.gp5"
        if not final_json.exists() or not report_json.exists() or not final_mid.exists():
            return _error(
                "The conversion finished without its expected artifacts.",
                500,
                f"Missing one of final.json, report.json, final.mid in {out_dir}",
            )
        try:
            records = _load_notes(final_json)
            report = _load_json(report_json)
            exported = build_legacy_gp5(
                records,
                final_mid,
                gp5_path,
                capo=int(report.get("capo", 0)),
                title=title,
            )
        except Exception as exc:  # noqa: BLE001 - surface a useful local error
            return _error("The Guitar Pro tab could not be written.", 500, str(exc))

    try:
        records = _load_notes(final_json)
        report = _load_json(report_json)
    except (OSError, ValueError) as exc:
        return _error("The conversion artifacts could not be read.", 500, str(exc))

    capo = int(report.get("capo", 0))
    summary = _summary(report, records, exported, pipeline)
    files = {
        "audio": audio_name,
        "midi": midi_name,
        "title": title,
    }
    report_payload = dict(report)
    report_payload["summary"] = summary
    report_payload["files"] = files
    if pipeline is not None:
        report_payload["pipeline"] = pipeline
    try:
        (job_dir / "report.json").write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return _error("The job report could not be written.", 500, str(exc))

    elapsed = round(time.time() - started, 1)
    base = f"/api/jobs/{job_id}"
    urls = {
        "gp5": f"{base}/score.gp5",
        "midi": f"{base}/final.mid",
        "tab": f"{base}/final_tab.txt",
        "report": f"{base}/report.json",
    }
    if audio_mode:
        urls["audio"] = f"{base}/source.wav"
    return jsonify({
        "id": job_id,
        "title": title,
        "capo": capo,
        "elapsed": elapsed,
        "files": files,
        "summary": summary,
        "urls": urls,
    })


# Each logical file name can resolve to a new-pipeline artifact or a legacy
# MIDI-only job. The first existing path wins.
_ALLOWED_FILES = {
    "score.gp5": (("score.gp5", "out/song.gp5"), "application/x-guitar-pro"),
    "final.mid": (("out/final.mid", "out/tabula/final.mid"), "audio/midi"),
    "final_tab.txt": (("out/final_tab.txt", "out/tab_with_features.txt"), "text/plain; charset=utf-8"),
    "report.json": (("report.json", "out/report.json"), "application/json"),
    "upload.mid": (("transcribed.mid", "transcribed.midi", "upload.mid"), "audio/midi"),
    "source.wav": (("source.wav",), "audio/wav"),
}


@app.get("/api/jobs/<job_id>/<name>")
def job_file(job_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    allowed = _ALLOWED_FILES.get(name)
    if allowed is None:
        abort(404)
    relative_names, mime = allowed
    job_dir = JOBS / job_id
    for relative in relative_names:
        path = job_dir / relative
        if path.is_file():
            return send_file(
                path,
                mimetype=mime,
                as_attachment=name != "score.gp5",
                download_name=name,
                conditional=True,
            )
    abort(404)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({
        "error": (
            f"That take is larger than the {MAX_UPLOAD_MB} MB upload limit. "
            f"Keep the recording under {MAX_AUDIO_MB} MB and the MIDI under "
            f"{MAX_MIDI_MB} MB."
        ),
    }), 413


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found."}), 404
    return "Not found", 404


def main() -> int:
    parser = argparse.ArgumentParser(description="Local audio + MIDI to tab web app")
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
