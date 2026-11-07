"""FluidSynth-based guitar synthesis using a General MIDI SoundFont.

Limits render duration to the actual MIDI length plus a short release tail.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi

from core.quantise import PPQ, ticks_to_seconds  # noqa: E402

def _resolve_default_sf2() -> str:
    """Pick the SoundFont used for synthesis.

    Priority: ``$TABULA_SF2``, then the Live HQ Natural GM V3 SoundFont under
    ``~/soundfonts``, then the distro FluidR3_GM bank.  The environment
    override keeps the choice machine-independent; the fallback keeps the
    package usable on hosts that do not have the large bank installed.
    """
    env = os.environ.get("TABULA_SF2")
    if env:
        return env
    live_hq = Path.home() / "soundfonts" / "Live_HQ_Natural_GM_V3.sf2"
    if live_hq.exists():
        return str(live_hq)
    return "/usr/share/sounds/sf2/FluidR3_GM.sf2"


# Live HQ Natural GM V3 by default (see musical-artifacts.com/artifacts/1176).
DEFAULT_SF2 = _resolve_default_sf2()


def _to_dict(note) -> dict:
    """Convert note-like object to dict with t, dur, pitch, s, f keys."""
    # numpy structured array record
    if hasattr(note, "dtype") and hasattr(note, "tolist"):
        return {name: note[name] for name in note.dtype.names}
    # dict-like
    if hasattr(note, "get"):
        return {k: note.get(k) for k in ("t", "dur", "pitch", "s", "f")}
    # Note object
    if hasattr(note, "t"):
        return {"t": note.t, "dur": note.dur, "pitch": note.pitch, "s": note.s, "f": note.f}
    # Fallback: assume sequence
    return {"t": note[0], "dur": note[1], "pitch": note[2], "s": note[3], "f": note[4]}


def _to_py(val):
    """Convert numpy scalars to native Python types."""
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


def _midi_duration_seconds(mid_path: Path) -> float:
    """Return the end time of the last note in seconds."""
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    return pm.get_end_time()


def _write_temp_midi(
    notes, bpm: float, ppq: int = PPQ,
    tempo_changes: list[tuple[float, float]] | None = None,
    source_midi_path: Path | None = None,
) -> Path:
    """Write notes to a temporary MIDI file and return its path.

    String 0 is a pitch-only input (a normal MIDI file carries no fingering);
    those notes must still be rendered.  Strings 1-6 are output fingering.
    If ``source_midi_path`` is given, that file is loaded and its instruments
    are replaced with our string tracks, preserving the original tempo map.
    Note times are converted from ticks to seconds using ``tempo_changes``.
    """
    if source_midi_path is not None:
        pm = pretty_midi.PrettyMIDI(str(source_midi_path))
        pm.instruments = []  # drop original instruments, keep tempo map etc.
    else:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    string_names = ["E2", "A2", "D3", "G3", "B3", "E4"]
    tracks = {}
    for s in [0, *range(1, 7)]:
        name = "pitch only" if s == 0 else f"string {s} ({string_names[s-1]})"
        track = pretty_midi.Instrument(program=24, name=name)  # GM nylon/classical
        pm.instruments.append(track)
        tracks[s] = track

    for n in notes:
        d = n  # already dict from synthesize
        s = int(d.get("s") or 0)
        if s not in tracks:
            s = 0
        pitch = _to_py(d.get("pitch"))
        start = ticks_to_seconds(int(d.get("t")), tempo_changes, ppq)
        end = ticks_to_seconds(int(d.get("t")) + max(1, int(d.get("dur"))), tempo_changes, ppq)
        tracks[s].notes.append(pretty_midi.Note(
            velocity=100, pitch=pitch,
            start=start, end=end
        ))

    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    pm.write(tmp.name)
    return Path(tmp.name)


def synthesize(
    notes,
    bpm: float,
    sr: int = 44100,
    ppq: int = PPQ,
    sf2_path: str = DEFAULT_SF2,
    release_tail: float = 0.5,
    tempo_changes: list[tuple[float, float]] | None = None,
    source_midi_path: Path | None = None,
) -> np.ndarray:
    """Render canonical notes to mono float32 audio via FluidSynth.

    Args:
        notes: Sequence of note-like objects (dict, Note, numpy record, etc.)
               with fields t, dur, pitch, s, f
        bpm: Tempo in beats per minute
        sr: Sample rate
        ppq: Ticks per quarter note
        sf2_path: Path to SoundFont (.sf2)
        release_tail: Extra seconds to render after last note for natural decay

    Returns:
        float32 mono audio array at sample rate sr
    """
    if not len(notes):
        return np.zeros(sr, dtype=np.float32)

    # Normalize to dicts with native Python values
    notes = [_to_dict(n) for n in notes]
    notes = [{k: _to_py(v) for k, v in d.items()} for d in notes]

    # Write notes to temporary MIDI
    mid_path = _write_temp_midi(notes, bpm, ppq, tempo_changes=tempo_changes,
                                source_midi_path=source_midi_path)

    try:
        # Get MIDI duration
        midi_dur = _midi_duration_seconds(mid_path)
        render_dur = midi_dur + release_tail
        max_samples = int(render_dur * sr)

        # Render with FluidSynth
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        result = subprocess.run([
            "fluidsynth", "-ni", sf2_path, str(mid_path),
            "-F", str(wav_path), "-r", str(sr),
            "-g", "1.0",      # gain
            "-R", "0",        # no reverb
            "-C", "0",        # no chorus
        ], capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"FluidSynth failed: {result.stderr}")

        # Read and trim to render duration.
        # FluidSynth writes stereo; downmix to mono before trimming so we
        # don't interpret interleaved L/R frames as twice as many samples.
        import wave
        with wave.open(str(wav_path), "rb") as w:
            nchan = w.getnchannels()
            sw = w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if sw == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width {sw}")
        if nchan > 1:
            audio = audio.reshape(-1, nchan).mean(axis=1)

        # Trim or pad to exact render duration
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        # Normalize peak
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio *= 0.89 / peak

        # Quick fade in/out
        fade = min(int(0.005 * sr), len(audio) // 2)
        if fade > 1:
            audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        return audio.astype(np.float32, copy=False)

    finally:
        # Cleanup temp files
        try:
            mid_path.unlink()
        except OSError:
            pass
        try:
            wav_path.unlink()
        except OSError:
            pass


def render_midi_file_to_wav(
    midi_path: str | Path,
    out_wav: str | Path,
    *,
    sr: int = 22050,
    sf2_path: str = DEFAULT_SF2,
    release_tail: float = 0.5,
) -> Path:
    """Render an existing MIDI file to mono WAV with FluidSynth.

    MuScriptor MIDI is shifted so bar 1 starts at tick 0.  For mixing with the
    original recording we subtract the ``muscriptor:bar_offset`` marker first,
    so both audio sources share the same timeline.
    """
    import mido
    import wave

    source = Path(midi_path)
    output = Path(out_wav)
    output.parent.mkdir(parents=True, exist_ok=True)

    offset = 0.0
    try:
        for track in mido.MidiFile(str(source)).tracks:
            for message in track:
                if (
                    message.type == "marker"
                    and str(message.text).startswith("muscriptor:bar_offset=")
                ):
                    offset = float(
                        str(message.text).split("muscriptor:bar_offset=", 1)[1]
                    )
                    break
    except Exception:
        offset = 0.0

    pm = pretty_midi.PrettyMIDI(str(source))
    for instrument in pm.instruments:
        for note in instrument.notes:
            note.start = max(0.0, float(note.start) - offset)
            note.end = max(note.start + 0.01, float(note.end) - offset)

    tmp_midi = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_midi_path = Path(tmp_midi.name)
    tmp_wav_path = Path(tmp_wav.name)
    tmp_midi.close()
    tmp_wav.close()
    try:
        pm.write(str(tmp_midi_path))
        render_duration = max(0.1, float(pm.get_end_time()) + float(release_tail))
        result = subprocess.run(
            [
                "fluidsynth", "-ni", sf2_path, str(tmp_midi_path),
                "-F", str(tmp_wav_path), "-r", str(sr),
                "-g", "1.0", "-R", "0", "-C", "0",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FluidSynth failed: {result.stderr}")
        with wave.open(str(tmp_wav_path), "rb") as handle:
            nchan = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width {width}")
        if nchan > 1:
            audio = audio.reshape(-1, nchan).mean(axis=1)
        max_samples = max(1, int(render_duration * sr))
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1e-6:
            audio *= 0.89 / peak
        import soundfile as sf

        sf.write(str(output), audio.astype(np.float32), sr, subtype="PCM_16")
        return output
    finally:
        for path in (tmp_midi_path, tmp_wav_path):
            try:
                path.unlink()
            except OSError:
                pass


def render_note_sample(
    pitch: int,
    out_wav: str | Path,
    *,
    program: int = 24,
    duration: float = 2.5,
    velocity: int = 96,
    sr: int = 22050,
    release_tail: float = 0.35,
) -> Path:
    """Render one monitor note to a mono WAV sample.

    The labeller schedules these per note so it can skip or mute individual
    notes without re-rendering the whole song.  ``program=24`` is GM nylon
    guitar; the dead-note thump uses ``program=28`` (muted electric guitar)
    with a very short duration.
    """
    pitch = max(0, min(127, int(pitch)))
    velocity = max(1, min(127, int(velocity)))
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=int(program), name="monitor")
    start = 0.0
    instrument.notes.append(
        pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + max(0.03, float(duration)),
        )
    )
    pm.instruments.append(instrument)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        pm.write(str(tmp_path))
        return render_midi_file_to_wav(
            tmp_path,
            out_wav,
            sr=int(sr),
            release_tail=float(release_tail),
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
