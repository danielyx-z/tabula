"""Local web app: source audio + MIDI transcription -> Guitar Pro tab."""

from __future__ import annotations

import argparse
import hashlib
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

_EARLY_REPO = Path(__file__).resolve().parents[1]
if str(_EARLY_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_EARLY_REPO / "src"))

try:
    from webapp.gp5 import build_gp5 as build_legacy_gp5
except ImportError:  # Allow ``python webapp/server.py`` as well as ``-m webapp.server``.
    from gp5 import build_gp5 as build_legacy_gp5

from transcribe import MuscriptorUnavailable, transcribe_audio
from tabfeatures.labeling.store import (
    LABEL_ROOT,
    append_labels,
    create_session,
    export_labels,
    label_state,
    list_sessions,
    mark_reviewed,
    remove_session,
    write_candidates,
    write_meta,
)
from tabfeatures.midi_groups import load_note_groups
from audio.synth import DEFAULT_SF2, render_midi_file_to_wav, render_note_sample

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
MUSCRIPTOR_MODEL = os.environ.get("TABULA_MUSCRIPTOR_MODEL", "small")
MUSCRIPTOR_INSTRUMENTS = os.environ.get(
    "TABULA_MUSCRIPTOR_INSTRUMENTS", "acoustic_guitar"
)
MUSCRIPTOR_GRID = int(os.environ.get("TABULA_MUSCRIPTOR_GRID", "8"))
MUSCRIPTOR_QUANTIZE = os.environ.get("TABULA_MUSCRIPTOR_QUANTIZE", "0") != "0"
MAX_CAPO = min(6, int(os.environ.get("TABULA_MAX_CAPO", "6")))

AUDIO_SUFFIXES = {
    ".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma",
}
MIDI_SUFFIXES = {".mid", ".midi"}

CONVERT_LOCK = threading.Lock()
JOBS.mkdir(parents=True, exist_ok=True)
MONITOR_SAMPLE_ROOT = _project_path(
    os.environ.get("TABULA_MONITOR_SAMPLES", "data/models/monitor_samples")
)
CACHE_ROOT = _project_path(os.environ.get("TABULA_CACHE", "data/cache"))
CACHE_VERSION = "v1"

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
    non_note = (pipeline.get("non_note") or {}).get("converted", 0)
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
        "non_note": int(non_note or 0),
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


def _local_transcribe(audio_path: Path, out_path: Path) -> None:
    """Run the local small MuScriptor model, preserving the tempo map."""
    transcribe_audio(
        audio_path,
        out_path,
        model=MUSCRIPTOR_MODEL,
        instrument=MUSCRIPTOR_INSTRUMENTS,
        quantize=MUSCRIPTOR_QUANTIZE,
        grid=MUSCRIPTOR_GRID,
        timeout=CONVERT_TIMEOUT,
    )


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


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _log_tail(path: Path, limit: int = 8000) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])[-limit:]
    except OSError:
        return ""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_file(kind: str, key: str, suffix: str) -> Path:
    return CACHE_ROOT / CACHE_VERSION / kind / f"{key}{suffix}"


def _cache_restore(cache_path: Path, target: Path) -> bool:
    if not cache_path.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(cache_path, target)
        return True
    except OSError:
        return False


def _cache_store(source: Path, cache_path: Path) -> None:
    if not source.is_file():
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cache_path)
    except OSError:
        pass


def _write_job_progress(job_dir: Path, payload: dict) -> None:
    path = job_dir / "out" / "tabula" / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _conversion_error_message(log: str, audio_mode: bool) -> str:
    lowered = log.lower()
    if "match rate" in lowered:
        return "The recording and MIDI do not line up."
    if audio_mode and ("audio" in lowered or "ffmpeg" in lowered):
        return "The recording could not be read."
    return "The tab could not be built from this take."


def _parse_capo(value: object) -> str:
    raw = str(value or "auto").strip().lower()
    if raw in {"auto", "off"}:
        return raw
    try:
        capo = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Capo must be Auto, Off, or a number from 0 to {MAX_CAPO}."
        ) from exc
    if not 0 <= capo <= MAX_CAPO:
        raise ValueError(f"Capo must be Auto, Off, or a number from 0 to {MAX_CAPO}.")
    return str(capo)


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


@app.get("/label")
def label_page():
    return render_template(
        "label.html",
        max_audio_mb=MAX_AUDIO_MB,
        max_midi_mb=MAX_MIDI_MB,
        muscriptor_model=MUSCRIPTOR_MODEL,
        label_root=str(LABEL_ROOT),
    )


def _label_session_or_404(session_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        abort(404)
    if not (LABEL_ROOT / session_id).is_dir():
        abort(404)


def _audio_duration(path: Path) -> float:
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


@app.get("/api/label/sessions")
def label_list():
    return jsonify({"sessions": list_sessions()})


@app.post("/api/label/sessions")
def label_create():
    audio_upload = request.files.get("audio")
    midi_upload = request.files.get("midi")
    if audio_upload is None or not audio_upload.filename:
        return _error("Choose a recording to label.")
    try:
        audio_original, audio_suffix = _validate_upload(
            audio_upload,
            label="label audio",
            suffixes=AUDIO_SUFFIXES,
            max_mb=MAX_AUDIO_MB,
        )
    except ValueError as exc:
        return _error(str(exc))

    midi_original = None
    if midi_upload is not None and midi_upload.filename:
        try:
            midi_original, _midi_suffix = _validate_upload(
                midi_upload,
                label="label MIDI",
                suffixes=MIDI_SUFFIXES,
                max_mb=MAX_MIDI_MB,
            )
        except ValueError as exc:
            return _error(str(exc))

    session_id, directory = create_session()
    upload_path = directory / f"upload_audio{audio_suffix}"
    audio_path = directory / "audio.wav"
    midi_path = directory / "transcribed.mid"
    try:
        audio_upload.save(upload_path)
        if not upload_path.stat().st_size:
            raise ValueError("That recording is empty.")
        normalized = _normalize_audio(upload_path, audio_path)
        if normalized != audio_path:
            shutil.copy2(normalized, audio_path)
        if midi_original is not None:
            midi_upload.save(midi_path)
            source = "uploaded"
        else:
            _local_transcribe(audio_path, midi_path)
            source = "muscriptor"
        if not midi_path.is_file() or not midi_path.stat().st_size:
            raise ValueError("No MIDI transcription was produced.")
        payload = load_note_groups(
            midi_path,
            duration_s=_audio_duration(audio_path),
        )
        write_candidates(session_id, payload)
        note_count = sum(len(group.get("notes", [])) for group in payload["groups"])
        write_meta(
            session_id,
            {
                "original_name": audio_original.name,
                "midi_name": midi_original.name if midi_original else "transcribed.mid",
                "source": source,
                "model": MUSCRIPTOR_MODEL if source == "muscriptor" else None,
                "quantize": MUSCRIPTOR_QUANTIZE if source == "muscriptor" else None,
                "grid": MUSCRIPTOR_GRID if source == "muscriptor" else None,
                "duration": payload.get("duration"),
                "note_count": note_count,
                "dead_count": 0,
                "noise_count": 0,
                "ignore_count": 0,
                "reviewed": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    except MuscriptorUnavailable as exc:
        remove_session(session_id)
        return _error("Local MuScriptor is not available.", 500, str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced to the local UI
        remove_session(session_id)
        return _error(
            "The label session could not be created.", 500, str(exc)
        )
    finally:
        upload_path.unlink(missing_ok=True)

    return jsonify(label_state(session_id))


@app.get("/api/label/sessions/<session_id>")
def label_get(session_id: str):
    _label_session_or_404(session_id)
    return jsonify(label_state(session_id))


def _monitor_sample_dir() -> Path:
    """Cache monitor samples by SoundFont identity."""
    sf2 = Path(DEFAULT_SF2)
    try:
        stat = sf2.stat()
        signature = f"monitor-v2:{sf2.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        signature = f"monitor-v2:{sf2}:missing"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return MONITOR_SAMPLE_ROOT / digest


def _monitor_sample_path(kind: str, pitch: int = 0) -> Path:
    directory = _monitor_sample_dir() / kind
    if kind == "dead":
        return directory / "dead.wav"
    return directory / f"{int(pitch)}.wav"


def _ensure_monitor_sample(kind: str, pitch: int = 0) -> Path:
    path = _monitor_sample_path(kind, pitch)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.wav")
    try:
        if kind == "dead":
            render_note_sample(
                45,
                tmp,
                program=28,
                duration=0.12,
                velocity=84,
                release_tail=0.12,
            )
        else:
            render_note_sample(
                int(pitch),
                tmp,
                program=24,
                duration=2.5,
                velocity=96,
                release_tail=0.35,
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@app.get("/api/monitor/samples/pitched/<int:pitch>.wav")
def monitor_pitched_sample(pitch: int):
    if pitch < 0 or pitch > 127:
        abort(404)
    try:
        path = _ensure_monitor_sample("pitched", pitch)
    except Exception as exc:  # noqa: BLE001 - optional monitoring aid
        return _error("The MIDI monitor sample could not be rendered.", 500, str(exc))
    return send_file(
        path,
        mimetype="audio/wav",
        conditional=True,
        max_age=31536000,
    )


@app.get("/api/monitor/samples/dead.wav")
def monitor_dead_sample():
    try:
        path = _ensure_monitor_sample("dead")
    except Exception as exc:  # noqa: BLE001 - optional monitoring aid
        return _error("The dead-note monitor sample could not be rendered.", 500, str(exc))
    return send_file(
        path,
        mimetype="audio/wav",
        conditional=True,
        max_age=31536000,
    )


@app.get("/api/label/sessions/<session_id>/synth")
def label_synth(session_id: str):
    _label_session_or_404(session_id)
    directory = LABEL_ROOT / session_id
    midi_path = directory / "transcribed.mid"
    synth_path = directory / "synth.wav"
    if not midi_path.is_file():
        abort(404)
    if not synth_path.is_file():
        try:
            render_midi_file_to_wav(midi_path, synth_path)
        except Exception as exc:  # noqa: BLE001 - optional monitoring aid
            return _error("The MIDI monitor track could not be rendered.", 500, str(exc))
    return send_file(synth_path, mimetype="audio/wav", conditional=True)


@app.get("/api/label/sessions/<session_id>/audio")
def label_audio(session_id: str):
    _label_session_or_404(session_id)
    path = LABEL_ROOT / session_id / "audio.wav"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.post("/api/label/sessions/<session_id>/labels")
def label_append(session_id: str):
    _label_session_or_404(session_id)
    payload = request.get_json(silent=True) or {}
    raw = payload.get("labels")
    if raw is None:
        raw = [payload]
    if not isinstance(raw, list):
        return _error("labels must be a list.")
    records = [item for item in raw if isinstance(item, dict)]
    append_labels(session_id, records, op_id=uuid.uuid4().hex)
    state = label_state(session_id)
    meta = state.get("meta") or {}
    summary = state.get("summary") or {}
    meta["dead_count"] = int(summary.get("dead", 0))
    meta["noise_count"] = int(summary.get("noise", 0))
    meta["ignore_count"] = int(summary.get("ignore", 0))
    write_meta(session_id, meta)
    return jsonify(state)


@app.post("/api/label/sessions/<session_id>/reviewed")
def label_reviewed(session_id: str):
    _label_session_or_404(session_id)
    payload = request.get_json(silent=True) or {}
    mark_reviewed(session_id, bool(payload.get("reviewed", True)))
    return jsonify(label_state(session_id))


@app.get("/api/label/sessions/<session_id>/labels.jsonl")
def label_export(session_id: str):
    _label_session_or_404(session_id)
    body = export_labels(session_id)
    return app.response_class(
        body,
        mimetype="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{session_id}-labels.jsonl"'
        },
    )


@app.delete("/api/label/sessions/<session_id>")
def label_delete(session_id: str):
    _label_session_or_404(session_id)
    remove_session(session_id)
    return jsonify({"ok": True})


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
    try:
        capo = _parse_capo(request.form.get("capo", "auto"))
    except ValueError as exc:
        return _error(str(exc))
    fast_mode = str(request.form.get("fast", "1")).strip().lower() in {
        "1", "true", "on", "yes"
    }
    beam = 1 if fast_mode else 4

    has_midi = bool(midi_upload and midi_upload.filename)
    has_audio = bool(audio_upload and audio_upload.filename)
    if not has_midi and not has_audio:
        return _error("Choose a recording, a MIDI transcription, or both.")

    midi_original = None
    midi_suffix = ".mid"
    if has_midi:
        try:
            midi_original, midi_suffix = _validate_upload(
                midi_upload,
                label="MIDI transcription",
                suffixes=MIDI_SUFFIXES,
                max_mb=MAX_MIDI_MB,
            )
        except ValueError as exc:
            return _error(str(exc))

    audio_mode = has_audio
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

    requested_job = str(request.form.get("job_id") or "").strip()
    if requested_job:
        if not re.fullmatch(r"[0-9a-f]{32}", requested_job):
            return _error("The job id is invalid.")
        if (JOBS / requested_job).exists():
            return _error("That job id is already in use.")
        job_id = requested_job
    else:
        job_id = uuid.uuid4().hex
    job_dir = JOBS / job_id
    out_dir = job_dir / "out"
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return _error("That job id is already in use.")
    _write_job_progress(
        job_dir,
        {
            "stage": "transcribe",
            "message": (
                "Preparing the uploaded MIDI…"
                if has_midi
                else "Transcribing the recording with MuScriptor…"
            ),
            "tab": "",
        },
    )

    midi_name = (
        midi_original.name
        if midi_original is not None
        else f"{Path(audio_original.name).stem}.mid"
    )
    midi_path = job_dir / "transcribed.mid"
    audio_path = None
    normalized_audio = None
    if audio_mode:
        audio_name = audio_original.name if audio_original else f"recording{audio_suffix}"
        audio_path = job_dir / f"upload_audio{audio_suffix}"
        normalized_audio = job_dir / "source.wav"
    else:
        audio_name = None

    detect_proc: subprocess.Popen | None = None
    detect_log = None
    detect_cache: Path | None = None
    detect_cached = False
    audio_digest = ""
    midi_cache: Path | None = None
    try:
        if audio_mode:
            audio_upload.save(audio_path)
            if audio_path.stat().st_size == 0:
                raise ValueError("That recording is empty.")
            if _size_mb(audio_path) > MAX_AUDIO_MB:
                raise ValueError(f"The source recording is larger than {MAX_AUDIO_MB} MB.")
            normalized_audio = _normalize_audio(audio_path, normalized_audio)
            audio_digest = _file_digest(normalized_audio)
            detect_signature = (
                f"{CACHE_VERSION}:{audio_digest}:"
                "sr=22050:delta=0.05:wait=2:before=0.03:after=0.18"
            )
            detect_key = hashlib.sha1(detect_signature.encode("utf-8")).hexdigest()
            detect_cache = _cache_file("audio_events", detect_key, ".json")
            if _cache_restore(detect_cache, out_dir / "audio_events.json"):
                detect_cached = True
            else:
                detect_log = (job_dir / "detect.log").open("w", encoding="utf-8")
                detect_proc = subprocess.Popen(
                    [
                        PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
                        "--audio", str(normalized_audio),
                        "--midi", str(midi_path),
                        "--out-dir", str(out_dir),
                        "--stage", "detect",
                    ],
                    cwd=str(REPO),
                    stdout=detect_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            _write_job_progress(
                job_dir,
                {
                    "stage": "audio",
                    "message": (
                        "Using cached audio analysis…"
                        if detect_cached
                        else (
                            "Detecting note attacks…"
                            if has_midi
                            else "Transcribing with MuScriptor and detecting note attacks…"
                        )
                    ),
                    "tab": "",
                },
            )
        if has_midi:
            midi_upload.save(midi_path)
            if midi_path.stat().st_size == 0:
                raise ValueError("That MIDI file is empty.")
            if _size_mb(midi_path) > MAX_MIDI_MB:
                raise ValueError(f"The MIDI transcription is larger than {MAX_MIDI_MB} MB.")
        else:
            midi_signature = (
                f"{CACHE_VERSION}:{audio_digest}:{MUSCRIPTOR_MODEL}:"
                f"quantize={int(MUSCRIPTOR_QUANTIZE)}:grid={MUSCRIPTOR_GRID}"
            )
            midi_key = hashlib.sha1(midi_signature.encode("utf-8")).hexdigest()
            midi_cache = _cache_file("muscriptor", midi_key, ".mid")
            if not _cache_restore(midi_cache, midi_path):
                try:
                    _local_transcribe(normalized_audio, midi_path)
                except MuscriptorUnavailable as exc:
                    raise MuscriptorUnavailable(str(exc)) from exc
                except RuntimeError as exc:
                    raise RuntimeError(f"MuScriptor transcription failed: {exc}") from exc
                _cache_store(midi_path, midi_cache)
    except MuscriptorUnavailable as exc:
        _stop_process(detect_proc)
        if detect_log is not None:
            detect_log.close()
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error("Local MuScriptor is not available.", 500, str(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        _stop_process(detect_proc)
        if detect_log is not None:
            detect_log.close()
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error(str(exc) or "The upload could not be saved.", 400)
    except Exception as exc:  # noqa: BLE001 - surfaced as a local error
        _stop_process(detect_proc)
        if detect_log is not None:
            detect_log.close()
        shutil.rmtree(job_dir, ignore_errors=True)
        return _error("The upload could not be saved.", 500, str(exc))

    title = Path(audio_name or midi_name).stem or "Untitled take"
    started = time.time()

    tab_proc: subprocess.Popen | None = None
    tab_log = None
    if audio_mode:
        tab_log = (job_dir / "tab.log").open("w", encoding="utf-8")
        tab_proc = subprocess.Popen(
            [
                PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
                "--audio", str(normalized_audio),
                "--midi", str(midi_path),
                "--out-dir", str(out_dir),
                "--stage", "tab",
                "--checkpoint", str(CHECKPOINT),
                "--vocab", str(VOCAB),
                "--beam", str(beam),
                "--capo", capo,
                "--max-capo", str(MAX_CAPO),
                "--force",
            ],
            cwd=str(REPO),
            stdout=tab_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Wait for the tab decoder first.  The draft score can be published
        # while the audio onset detector is still running.
        try:
            tab_rc = tab_proc.wait(timeout=CONVERT_TIMEOUT)
        except subprocess.TimeoutExpired:
            _stop_process(detect_proc)
            _stop_process(tab_proc)
            if detect_log is not None and not detect_log.closed:
                detect_log.close()
            if tab_log is not None:
                tab_log.close()
            return _error(
                "The conversion timed out.",
                504,
                f"The take did not finish within {CONVERT_TIMEOUT} seconds.",
            )
        finally:
            if tab_log is not None and not tab_log.closed:
                tab_log.close()
        if tab_rc != 0:
            _stop_process(detect_proc)
            if detect_log is not None and not detect_log.closed:
                detect_log.close()
            log = _log_tail(job_dir / "tab.log")
            return _error("The tab could not be built.", 500, log or f"Exit code {tab_rc}")

        try:
            progress_path = out_dir / "tabula" / "progress.json"
            progress = _load_json(progress_path) if progress_path.is_file() else {}
            if isinstance(progress, dict):
                progress["duration"] = round(float(_audio_duration(normalized_audio)), 3)
                _write_job_progress(job_dir, progress)
        except Exception:  # noqa: BLE001 - draft metadata is optional
            pass

        preview_log = (job_dir / "preview.log").open("w", encoding="utf-8")
        try:
            subprocess.run(
                [
                    PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
                    "--audio", str(normalized_audio),
                    "--midi", str(midi_path),
                    "--out-dir", str(out_dir),
                    "--stage", "provisional",
                    "--title", title,
                    "--capo", capo,
                ],
                cwd=str(REPO),
                stdout=preview_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=CONVERT_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            preview_log.close()

        # Only now wait for the audio detector; the draft score is already
        # available in progress.json.
        if detect_cached:
            detect_rc = 0
        else:
            try:
                detect_rc = detect_proc.wait(timeout=CONVERT_TIMEOUT) if detect_proc else 1
            except subprocess.TimeoutExpired:
                _stop_process(detect_proc)
                if detect_log is not None and not detect_log.closed:
                    detect_log.close()
                return _error(
                    "The conversion timed out.",
                    504,
                    f"The take did not finish within {CONVERT_TIMEOUT} seconds.",
                )
            finally:
                if detect_log is not None and not detect_log.closed:
                    detect_log.close()
        if detect_rc != 0:
            log = _log_tail(job_dir / "detect.log")
            return _error("Audio analysis failed.", 500, log or f"Exit code {detect_rc}")
        if detect_cache is not None and not detect_cached:
            _cache_store(out_dir / "audio_events.json", detect_cache)

        align_log = (job_dir / "align.log").open("w", encoding="utf-8")
        try:
            align_result = subprocess.run(
                [
                    PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
                    "--audio", str(normalized_audio),
                    "--midi", str(midi_path),
                    "--out-dir", str(out_dir),
                    "--stage", "align",
                ],
                cwd=str(REPO),
                stdout=align_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=CONVERT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            align_log.close()
            return _error(
                "The conversion timed out.",
                504,
                f"The take did not finish within {CONVERT_TIMEOUT} seconds.",
            )
        finally:
            if not align_log.closed:
                align_log.close()
        if align_result.returncode != 0:
            log = _log_tail(job_dir / "align.log")
            return _error(
                "Audio/MIDI alignment failed.",
                500,
                log or f"Exit code {align_result.returncode}",
            )

        stitch_log = (job_dir / "stitch.log").open("w", encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    PYTHON, str(REPO / "scripts" / "stitch_tab.py"),
                    "--audio", str(normalized_audio),
                    "--midi", str(midi_path),
                    "--out-dir", str(out_dir),
                    "--stage", "stitch",
                    "--title", title,
                    "--width", "120",
                    "--capo", capo,
                ],
                cwd=str(REPO),
                stdout=stitch_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=CONVERT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            stitch_log.close()
            return _error(
                "The conversion timed out.",
                504,
                f"The take did not finish within {CONVERT_TIMEOUT} seconds.",
            )
        finally:
            if not stitch_log.closed:
                stitch_log.close()
    else:
        command = [
            PYTHON, str(REPO / "scripts" / "midi2tab.py"), str(midi_path),
            "--checkpoint", str(CHECKPOINT),
            "--vocab", str(VOCAB),
            "--out-dir", str(out_dir),
            "--beam", str(beam),
            "--max-fret", "6",
            "--capo", capo,
            "--max-capo", str(MAX_CAPO),
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
    "progress.json": (("out/tabula/progress.json",), "application/json"),
    "provisional.gp5": (("out/provisional.gp5",), "application/x-guitar-pro"),
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
