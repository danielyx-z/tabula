"""Plain-assert tests for chord-safe chunking: ``python tests/test_chunk.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.fretboard import STANDARD  # noqa: E402
from core.schema import Note  # noqa: E402
from model.chunk import (  # noqa: E402
    ChunkConfig,
    _avoid_chord_split,
    make_inference_chunks,
    make_training_chunks,
)
from tablature.tokenizer import (  # noqa: E402
    DEFAULT_VOCAB, EOS_TOKEN, UNK_TOKEN, parse_tab,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def note(t: int, s: int) -> Note:
    return Note(t=t, dur=240, pitch=STANDARD[s - 1], s=s, f=0)


def no_split_boundaries(notes: list[Note], chunks) -> bool:
    for chunk in chunks:
        start = chunk.meta["note_start"]
        end = chunk.meta["note_end"]
        if start > 0 and notes[start - 1].t == notes[start].t:
            return False
        if end < len(notes) and notes[end - 1].t == notes[end].t:
            return False
    return True


def test_helper() -> None:
    notes = [note(0, 1), note(240, 2), note(480, 1), note(480, 2), note(480, 3), note(720, 4)]
    # Boundary at index 4 sits inside the t=480 chord -> move back to index 2.
    check("boundary retreats to chord start", _avoid_chord_split(notes, 0, 4, 6) == 2,
          str(_avoid_chord_split(notes, 0, 4, 6)))
    # If the chunk begins with the chord, extend through it instead.
    check("boundary extends through chord", _avoid_chord_split(notes, 2, 4, 6) == 5,
          str(_avoid_chord_split(notes, 2, 4, 6)))
    check("clean boundary untouched", _avoid_chord_split(notes, 0, 2, 6) == 2)


def test_training() -> None:
    notes = [note(0, 1), note(240, 2), note(480, 1), note(480, 2), note(480, 3), note(720, 4)]

    # Forced note-count mode must still respect chord edges.
    chunks = make_training_chunks(notes, DEFAULT_VOCAB, "x", ChunkConfig(training_chunk_notes=2))
    check("training chunks cover all notes",
          sum(c.meta["num_notes"] for c in chunks) == len(notes),
          str([c.meta["num_notes"] for c in chunks]))
    check("training chunks keep chords intact", no_split_boundaries(notes, chunks),
          str([(c.meta["note_start"], c.meta["note_end"]) for c in chunks]))

    # Token-cap mode (the real training path) also keeps chords intact.
    chunks = make_training_chunks(notes, DEFAULT_VOCAB, "x", ChunkConfig(max_tokens=40))
    check("token-capped training keeps chords intact", no_split_boundaries(notes, chunks),
          str([(c.meta["note_start"], c.meta["note_end"]) for c in chunks]))


def test_inference() -> None:
    notes = [note(0, 1), note(240, 2), note(480, 1), note(480, 2), note(480, 3), note(720, 4)]

    chunks = make_inference_chunks(notes, DEFAULT_VOCAB, "x", ChunkConfig(k=3))
    check("inference chunks keep chords intact", no_split_boundaries(notes, chunks),
          str([(c.meta["note_start"], c.meta["note_end"]) for c in chunks]))
    check("inference chunks cover all notes",
          sum(c.meta["num_notes"] for c in chunks) == len(notes))

    # A 2-note window that begins on the chord must grow to include all of it.
    chunks = make_inference_chunks(notes, DEFAULT_VOCAB, "x", ChunkConfig(k=2))
    check("short window grows through chord", no_split_boundaries(notes, chunks),
          str([(c.meta["note_start"], c.meta["note_end"]) for c in chunks]))
    check("chord chunk reports 3 notes", chunks[1].meta["num_notes"] == 3,
          str(chunks[1].meta["num_notes"]))


def test_chord_string_order() -> None:
    """Decoder emits TAB tokens within a chord onset in strictly ascending
    string order (s=1 thickest/low E -> s=6 thinnest/high E), regardless of
    the order the notes were supplied in."""
    # Same onset t=960, strings given out of order: 6, 3, 1, 4.
    raw = [
        note(960, 6), note(960, 3), note(960, 1), note(960, 4),
        note(1200, 2),  # next onset, single string
    ]
    vocab = DEFAULT_VOCAB
    chunks = make_training_chunks(raw, vocab, "x", ChunkConfig(max_tokens=256))

    def onsets_from_chunk(c):
        onsets, cur = [], []
        for i in c.dec_tokens:
            tok = vocab.id_to_token.get(i, UNK_TOKEN)
            if tok == EOS_TOKEN:
                continue
            if tok.startswith("TIME_SHIFT<"):
                if cur:
                    onsets.append(cur); cur = []
            elif tok.startswith("TAB<"):
                cur.append(parse_tab(tok))
        if cur:
            onsets.append(cur)
        return onsets

    saw_chord = False
    for c in chunks:
        for onset in onsets_from_chunk(c):
            if len(onset) < 2:
                continue
            strings = [s for s, f in onset]
            saw_chord = True
            check("chord onset strings strictly ascending",
                  strings == sorted(strings) and len(set(strings)) == len(strings),
                  str(strings))
    check("saw the multi-note chord onset", saw_chord)


def main() -> int:
    test_helper()
    test_training()
    test_inference()
    test_chord_string_order()
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
