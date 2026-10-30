"""Metrics: pitch accuracy, tab accuracy, difficulty, error decomposition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from core.fretboard import STANDARD, positions, pitch_of
from core.schema import Note
from .tokenizer import Vocab, parse_tab


@dataclass
class EvalCounts:
    """Aggregated counts for metric computation."""
    total_notes: int = 0
    pitch_correct: int = 0
    tab_correct: int = 0
    valid_predictions: int = 0  # predicted (s,f) is on fretboard

    # Error decomposition
    timing_errors: int = 0
    pitch_errors: int = 0
    position_errors: int = 0

    # Boundary vs interior
    boundary_total: int = 0
    boundary_correct: int = 0
    interior_total: int = 0
    interior_correct: int = 0

    # Playability violations
    same_string_overlap: int = 0
    chord_span_gt_5: int = 0
    simult_gt_4: int = 0

    # Playability difficulty (paper Eq. 1-6), accumulated over transitions
    difficulty_sum: float = 0.0
    difficulty_n: int = 0

    def pitch_accuracy(self) -> float:
        return self.pitch_correct / self.total_notes if self.total_notes else 0.0

    def tab_accuracy(self) -> float:
        return self.tab_correct / self.total_notes if self.total_notes else 0.0

    def validity(self) -> float:
        return self.valid_predictions / self.total_notes if self.total_notes else 0.0

    def boundary_accuracy(self) -> float:
        return self.boundary_correct / self.boundary_total if self.boundary_total else 0.0

    def interior_accuracy(self) -> float:
        return self.interior_correct / self.interior_total if self.interior_total else 0.0

    def difficulty(self) -> float:
        return self.difficulty_sum / self.difficulty_n if self.difficulty_n else 0.0


def decode_predictions(dec_ids: list[int], vocab: Vocab) -> list[tuple[int, int]]:
    """Decode decoder token IDs to (string, fret) pairs, ignoring non-TAB tokens."""
    tokens = vocab.decode(dec_ids)
    tabs = []
    for tok in tokens:
        if tok.startswith("TAB<"):
            s, f = parse_tab(tok)
            tabs.append((s, f))
    return tabs


def align_predictions(pred_tabs: list[tuple[int, int]],
                      gt_notes: list[Note],
                      is_boundary: list[bool] | None = None) -> EvalCounts:
    """
    Align predicted tabs to ground truth notes and compute metrics.
    Assumes pred_tabs and gt_notes are in the same order (onset, pitch).
    """
    counts = EvalCounts()
    n = min(len(pred_tabs), len(gt_notes))
    counts.total_notes = len(gt_notes)

    for i in range(n):
        pred_s, pred_f = pred_tabs[i]
        gt = gt_notes[i]

        # Validity: is prediction on fretboard?
        valid = 1 <= pred_s <= 6 and 0 <= pred_f <= 24
        if valid:
            counts.valid_predictions += 1

        # Pitch accuracy: does predicted (s,f) produce the right pitch?
        if valid and pitch_of(pred_s, pred_f, STANDARD) == gt.pitch:
            counts.pitch_correct += 1

        # Tab accuracy: exact (s,f) match
        if pred_s == gt.s and pred_f == gt.f:
            counts.tab_correct += 1
        else:
            # Error decomposition
            pred_pitch = pitch_of(pred_s, pred_f, STANDARD) if valid else -1
            if pred_pitch != gt.pitch:
                counts.pitch_errors += 1
            elif pred_s != gt.s or pred_f != gt.f:
                counts.position_errors += 1
            else:
                counts.timing_errors += 1  # shouldn't happen if aligned

        # Boundary vs interior
        if is_boundary is not None and i < len(is_boundary):
            if is_boundary[i]:
                counts.boundary_total += 1
                if pred_s == gt.s and pred_f == gt.f:
                    counts.boundary_correct += 1
            else:
                counts.interior_total += 1
                if pred_s == gt.s and pred_f == gt.f:
                    counts.interior_correct += 1

    return counts


def difficulty_terms(notes, alpha: float = 0.25) -> list[float]:
    """Per-transition difficulty from the paper's Eq. 1-6.

    ``notes`` may be ``Note`` or ``DecodedNote`` objects.  Positions are ordered
    by onset and string, and each consecutive pair contributes one term.
    """
    if len(notes) < 2:
        return []

    ordered = sorted(notes, key=lambda n: (n.t, n.s, n.f))
    terms: list[float] = []
    for p, q in zip(ordered, ordered[1:]):
        delta_fret = q.f - p.f
        if delta_fret > 0:
            fret_stretch = 0.50 * abs(delta_fret)
        else:
            fret_stretch = 0.75 * abs(delta_fret)

        locality = alpha * (p.f + q.f)

        # The paper writes Δstring = q - p.  Using the absolute value keeps the
        # metric symmetric and agrees with the stated 0..18.5 range.
        delta_string = abs(q.s - p.s)
        vertical_stretch = 0.25 if delta_string <= 1 else 0.50

        terms.append(fret_stretch + locality + vertical_stretch)
    return terms


def compute_difficulty(notes, alpha: float = 0.25) -> float:
    """Mean difficulty over all positions (paper Section 3.6)."""
    terms = difficulty_terms(notes, alpha)
    return float(np.mean(terms)) if terms else 0.0


def check_playability(pred_tabs: list[tuple[int, int]],
                      gt_notes: list[Note]) -> tuple[int, int, int]:
    """Check playability violations: same-string overlap, chord span > 5, >4 simultaneous."""
    from itertools import groupby

    same_string_overlap = 0
    chord_span_gt_5 = 0
    simult_gt_4 = 0

    # Group by onset
    for _, group in groupby(zip(pred_tabs, gt_notes), key=lambda x: x[1].t):
        chord = list(group)
        if len(chord) > 4:
            simult_gt_4 += 1

        strings = [c[0][0] for c in chord]
        frets = [c[0][1] for c in chord]

        if max(frets) - min(frets) > 5:
            chord_span_gt_5 += 1

        # Check same-string overlap (simplified: same string in chord)
        if len(set(strings)) < len(strings):
            same_string_overlap += 1

    return same_string_overlap, chord_span_gt_5, simult_gt_4


def aggregate_counts(all_counts: list[EvalCounts]) -> EvalCounts:
    """Sum multiple EvalCounts."""
    total = EvalCounts()
    for c in all_counts:
        total = EvalCounts(
            total_notes=total.total_notes + c.total_notes,
            pitch_correct=total.pitch_correct + c.pitch_correct,
            tab_correct=total.tab_correct + c.tab_correct,
            valid_predictions=total.valid_predictions + c.valid_predictions,
            timing_errors=total.timing_errors + c.timing_errors,
            pitch_errors=total.pitch_errors + c.pitch_errors,
            position_errors=total.position_errors + c.position_errors,
            boundary_total=total.boundary_total + c.boundary_total,
            boundary_correct=total.boundary_correct + c.boundary_correct,
            interior_total=total.interior_total + c.interior_total,
            interior_correct=total.interior_correct + c.interior_correct,
            same_string_overlap=total.same_string_overlap + c.same_string_overlap,
            chord_span_gt_5=total.chord_span_gt_5 + c.chord_span_gt_5,
            simult_gt_4=total.simult_gt_4 + c.simult_gt_4,
            difficulty_sum=total.difficulty_sum + c.difficulty_sum,
            difficulty_n=total.difficulty_n + c.difficulty_n,
        )
    return total


def print_metrics(counts: EvalCounts, prefix: str = "") -> None:
    """Pretty print metrics."""
    print(f"{prefix}Pitch Accuracy:   {counts.pitch_accuracy():.4f}")
    print(f"{prefix}Tab Accuracy:     {counts.tab_accuracy():.4f}")
    print(f"{prefix}Validity:         {counts.validity():.4f}")
    print(f"{prefix}Difficulty:       {counts.difficulty():.4f}")
    print(f"{prefix}Boundary Acc:     {counts.boundary_accuracy():.4f} ({counts.boundary_correct}/{counts.boundary_total})")
    print(f"{prefix}Interior Acc:     {counts.interior_accuracy():.4f} ({counts.interior_correct}/{counts.interior_total})")
    print(f"{prefix}Error Decomp:     timing={counts.timing_errors} pitch={counts.pitch_errors} position={counts.position_errors}")
    print(f"{prefix}Playability:      overlap={counts.same_string_overlap} span>5={counts.chord_span_gt_5} simult>4={counts.simult_gt_4}")
    print(f"{prefix}Total Notes:      {counts.total_notes}")