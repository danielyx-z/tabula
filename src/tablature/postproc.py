"""Post-processing from the Fretting-Transformer paper.

Section 3.5 defines two corrections:

1. **Overlap correction** — notes that land on the same string at overlapping
   times are moved to another valid position.  The input pitch is used, so this
   also repairs pitch errors on those notes.
2. **Neighbor search** — each input note is matched to the closest estimated
   note within ±5 notes.  If the pitch differs, the fingering is corrected; if
   no candidate exists, the first viable string/fret for the input pitch is
   used.  This makes pitch accuracy 100% after post-processing, as reported in
   Table 2 of the paper.  Because the fret is rewritten from the input pitch,
post-processed tab accuracy mostly measures the predicted *string*, not the
fret.

Every output note records whether its position came from a model TAB token
(``from_prediction``) or had to be invented by the ``_first_valid`` fallback,
so evaluation can report coverage and score only genuinely predicted notes.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.fretboard import STANDARD, positions, pitch_of
from core.schema import Note
from .tokenizer import parse_tab


@dataclass
class DecodedNote:
    """A tablature note with timing and a fingering."""

    t: int
    dur: int
    pitch: int
    s: int
    f: int
    is_boundary: bool = False
    # False when no model TAB token backs this position (padding or the
    # _first_valid fallback inside neighbor_search).
    from_prediction: bool = True


def _first_valid(
    pitch: int,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    capo: int = 0,
) -> tuple[int, int]:
    """Canonical fallback position: lowest fret, then lowest string.

    ``positions`` orders candidates by string (thickest first), so its first
    entry can be a high-fret position on the low E string even when an open
    string is available.  The paper's baseline uses the lowest possible fret
    for a pitch; use the same rule for the post-processing fallback.
    """
    candidates = positions(pitch, tuning, frets, capo)
    if candidates:
        return min(candidates, key=lambda sf: (sf[1], sf[0]))
    # Should only happen for out-of-range corpus notes; keep a valid-looking
    # placeholder so downstream code cannot crash.
    return 1, max(0, min(frets, pitch - tuning[0]))


def decode_to_notes(
    dec_ids: list[int],
    vocab,
    enc_notes: list[Note],
    tuning: tuple[int, ...] = STANDARD,
) -> list[DecodedNote]:
    """Decode TAB ids and align them to ``enc_notes`` (compatibility helper)."""
    tabs: list[tuple[int, int]] = []
    for tok_id in dec_ids:
        tok = vocab.id_to_token.get(tok_id, "")
        if tok.startswith("TAB<"):
            tabs.append(parse_tab(tok))
    return postprocess(tabs, enc_notes, mode="full", tuning=tuning)


def _overlaps(
    assigned: dict[int, list[DecodedNote]],
    string: int,
    start: int,
    end: int,
) -> bool:
    for other in assigned.get(string, []):
        other_end = other.t + max(1, other.dur)
        if start < other_end and other.t < end:
            return True
    return False


def repair_overlaps(
    notes: list[DecodedNote],
    gt_notes: list[Note] | None = None,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
    max_fret: int | None = None,
    capo: int = 0,
) -> list[DecodedNote]:
    """Move same-string overlapping notes to another valid position.

    When ``gt_notes`` is supplied, the move targets the input note's pitch, so
    the correction never changes the music.
    """
    if not notes:
        return []

    target_pitches: list[int] = []
    for i, note in enumerate(notes):
        if gt_notes is not None and i < len(gt_notes):
            target_pitches.append(gt_notes[i].pitch)
        else:
            target_pitches.append(note.pitch)

    assigned: dict[int, list[DecodedNote]] = {
        s: [] for s in range(1, len(tuning) + 1)
    }
    out: list[DecodedNote] = list(notes)

    order = sorted(
        range(len(notes)),
        key=lambda i: (notes[i].t, notes[i].s, notes[i].f),
    )
    for i in order:
        note = out[i]
        target = target_pitches[i]
        start = note.t
        end = note.t + max(1, note.dur)

        if not _overlaps(assigned, note.s, start, end):
            assigned[note.s].append(note)
            continue

        candidates = [
            (s2, f2)
            for s2, f2 in positions(target, tuning, frets, capo)
            if max_fret is None or f2 <= max_fret
        ]
        candidates = sorted(
            candidates,
            key=lambda sf: (
                abs(sf[0] - note.s),
                abs(sf[1] - note.f),
                0 if sf[1] == 0 else 1,
            ),
        )
        chosen: tuple[int, int] | None = None
        for s2, f2 in candidates:
            if not _overlaps(assigned, s2, start, end):
                chosen = (s2, f2)
                break

        if chosen is None:
            # No free position for this pitch; keep the note and accept the
            # overlap rather than inventing a wrong pitch.
            assigned[note.s].append(note)
            continue

        s2, f2 = chosen
        fixed = DecodedNote(
            t=note.t,
            dur=note.dur,
            pitch=target,
            s=s2,
            f=f2,
            is_boundary=note.is_boundary,
            from_prediction=note.from_prediction,
        )
        out[i] = fixed
        assigned[s2].append(fixed)

    return out


def neighbor_search(
    pred_notes: list[DecodedNote],
    gt_notes: list[Note],
    window: int = 5,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
) -> list[DecodedNote]:
    """Match every input note to the best estimated note within ±window."""
    if not gt_notes:
        return []

    used: set[int] = set()
    out: list[DecodedNote] = []

    for i, gt in enumerate(gt_notes):
        lo = max(0, i - window)
        hi = min(len(pred_notes), i + window + 1)

        best_j: int | None = None
        best_key: tuple[int, int] | None = None
        for j in range(lo, hi):
            if j in used:
                continue
            candidate = pred_notes[j]
            if not candidate.from_prediction:
                # Padded notes are not model estimates; matching them would
                # make the fallback look like a prediction.
                continue
            # Match by sequence position first.  Pitch-first matching lets a
            # single wrong prediction steal its neighbour's correct candidate,
            # which can cascade through a whole chunk and destroys otherwise
            # correct fingerings.
            key = (abs(j - i), abs(candidate.pitch - gt.pitch))
            if best_key is None or key < best_key:
                best_key = key
                best_j = j

        from_prediction = best_j is not None
        if best_j is None:
            s, f = _first_valid(gt.pitch, tuning, frets)
        else:
            used.add(best_j)
            candidate = pred_notes[best_j]
            from_prediction = candidate.from_prediction
            if candidate.pitch == gt.pitch:
                s, f = candidate.s, candidate.f
            else:
                # Prefer keeping the predicted string if it can sound the pitch.
                fret_on_string = gt.pitch - tuning[candidate.s - 1]
                if 0 <= fret_on_string <= frets:
                    s, f = candidate.s, fret_on_string
                else:
                    s, f = _first_valid(gt.pitch, tuning, frets)
                    from_prediction = False

        out.append(
            DecodedNote(
                t=gt.t,
                dur=gt.dur,
                pitch=gt.pitch,
                s=s,
                f=f,
                from_prediction=from_prediction,
            )
        )

    return out


def postprocess(
    pred,
    gt_notes: list[Note],
    mode: str = "full",
    window: int = 5,
    tuning: tuple[int, ...] = STANDARD,
    frets: int = 24,
) -> list[DecodedNote]:
    """Run the paper's post-processing pipeline.

    ``pred`` may be a list of ``(string, fret)`` pairs or ``DecodedNote``
    objects.  ``mode`` is ``"none"``, ``"overlap"`` or ``"full"``.
    """
    if mode not in ("none", "overlap", "full"):
        raise ValueError(f"unknown post-processing mode: {mode!r}")
    if not gt_notes:
        return []

    if pred and isinstance(pred[0], DecodedNote):
        tabs = [(n.s, n.f) for n in pred]
    else:
        tabs = list(pred)

    total = max(len(tabs), len(gt_notes))
    initial: list[DecodedNote] = []
    for i in range(total):
        if i < len(gt_notes):
            gt = gt_notes[i]
            t, dur = gt.t, gt.dur
            target_pitch = gt.pitch
        else:
            last = gt_notes[-1]
            t = last.t + (i - len(gt_notes) + 1) * 60
            dur = 60
            target_pitch = None

        if i < len(tabs):
            s, f = tabs[i]
            pitch = pitch_of(s, f, tuning)
        else:
            assert target_pitch is not None
            s, f = _first_valid(target_pitch, tuning, frets)
            pitch = target_pitch

        initial.append(
            DecodedNote(
                t=t,
                dur=dur,
                pitch=pitch,
                s=s,
                f=f,
                from_prediction=i < len(tabs),
            )
        )

    if mode == "none":
        # True raw output: do not invent notes that the model did not predict.
        return initial[: min(len(tabs), len(gt_notes))]

    notes = (
        repair_overlaps(initial, gt_notes, tuning, frets)
        if mode in ("overlap", "full")
        else initial
    )
    if mode == "full":
        notes = neighbor_search(notes, gt_notes, window, tuning, frets)
        # Neighbor search preserves pitch; a final overlap pass keeps the
        # fingering playable without changing the music.
        notes = repair_overlaps(notes, gt_notes, tuning, frets)

    return notes[: len(gt_notes)]
