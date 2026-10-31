"""Metrics: pitch accuracy, tab accuracy, difficulty, error decomposition, rhythm."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from core.fretboard import STANDARD, positions, pitch_of
from core.schema import Note
from .tokenizer import Vocab, parse_tab

# Fret above which a position is considered a high-position outlier.  GuitarSet
# has 0.23% of notes at fret >= 15; DadaGP has 2.12%, so this threshold makes
# the domain shift visible in the raw diagnostics.
HIGH_FRET = 15


@dataclass
class EvalCounts:
    """Aggregated counts for metric computation."""
    total_notes: int = 0
    pitch_correct: int = 0
    tab_correct: int = 0
    valid_predictions: int = 0  # predicted (s,f) is on fretboard

    # Coverage auditing: notes that received a genuine model TAB token, not a
    # post-processing fallback.  For raw counts this is min(len(pred), len(gt)).
    covered_notes: int = 0
    covered_tab_correct: int = 0

    # Error decomposition (removed dead timing_errors)
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

    # Fret/string diagnostics.  ``fret_hist`` counts every predicted TAB with a
    # fret in 0..24, including predictions whose pitch is wrong, so the
    # high-fret rate is an honest description of the decoder's output.
    pred_fret_sum: int = 0
    pred_fret_count: int = 0
    pred_string_correct: int = 0
    pred_fret_correct: int = 0
    fret_hist: list[int] = field(default_factory=lambda: [0] * 25)

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

    def coverage(self) -> float:
        """Fraction of ground-truth notes backed by a real model prediction."""
        return self.covered_notes / self.total_notes if self.total_notes else 0.0

    def tab_accuracy_at_pred(self) -> float:
        """Exact tab match over covered notes only (ignores fallbacks)."""
        return (
            self.covered_tab_correct / self.covered_notes
            if self.covered_notes
            else 0.0
        )

    def mean_fret(self) -> float:
        """Mean predicted fret over all predicted TAB positions."""
        return (
            self.pred_fret_sum / self.pred_fret_count
            if self.pred_fret_count
            else 0.0
        )

    def high_fret_count(self, threshold: int = HIGH_FRET) -> int:
        return sum(self.fret_hist[threshold:])

    def high_fret_rate(self, threshold: int = HIGH_FRET) -> float:
        """Fraction of predicted TAB positions at or above ``threshold``."""
        return (
            self.high_fret_count(threshold) / self.pred_fret_count
            if self.pred_fret_count
            else 0.0
        )

    def string_accuracy(self) -> float:
        """Fraction of GT slots whose predicted string is correct."""
        return (
            self.pred_string_correct / self.total_notes
            if self.total_notes
            else 0.0
        )

    def fret_accuracy(self) -> float:
        """Fraction of GT slots whose predicted fret is correct."""
        return (
            self.pred_fret_correct / self.total_notes
            if self.total_notes
            else 0.0
        )


@dataclass
class RhythmCounts:
    """Rhythm metrics computed from decoded timed notes."""
    matched: int = 0
    missing: int = 0
    extra: int = 0
    onset_errors: list[int] = None  # list of |Δt| in ticks for matched notes
    ioi_errors: list[int] = None    # list of |ΔIOI| in ticks
    dur_errors: list[int] = None    # list of |Δdur| in ticks

    def __post_init__(self):
        if self.onset_errors is None:
            self.onset_errors = []
        if self.ioi_errors is None:
            self.ioi_errors = []
        if self.dur_errors is None:
            self.dur_errors = []

    def onset_hit_rate(self, tolerance: int) -> float:
        if not self.onset_errors:
            return 0.0
        return sum(1 for e in self.onset_errors if e <= tolerance) / len(self.onset_errors)

    def median_onset_error(self) -> float:
        if not self.onset_errors:
            return 0.0
        return float(np.median(self.onset_errors))

    def mean_onset_error(self) -> float:
        if not self.onset_errors:
            return 0.0
        return float(np.mean(self.onset_errors))

    def median_ioi_error(self) -> float:
        if not self.ioi_errors:
            return 0.0
        return float(np.median(self.ioi_errors))

    def mean_ioi_error(self) -> float:
        if not self.ioi_errors:
            return 0.0
        return float(np.mean(self.ioi_errors))

    def median_dur_error(self) -> float:
        if not self.dur_errors:
            return 0.0
        return float(np.median(self.dur_errors))

    def mean_dur_error(self) -> float:
        if not self.dur_errors:
            return 0.0
        return float(np.mean(self.dur_errors))

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "missing": self.missing,
            "extra": self.extra,
            "onset_hit_rate_20": self.onset_hit_rate(20),
            "onset_hit_rate_60": self.onset_hit_rate(60),
            "onset_hit_rate_120": self.onset_hit_rate(120),
            "median_onset_error": self.median_onset_error(),
            "mean_onset_error": self.mean_onset_error(),
            "median_ioi_error": self.median_ioi_error(),
            "mean_ioi_error": self.mean_ioi_error(),
            "median_dur_error": self.median_dur_error(),
            "mean_dur_error": self.mean_dur_error(),
        }


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
    Assumes pred_tabs and gt_notes are in the same order (onset, string, pitch).
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

        # Fret/string diagnostics, independent of pitch correctness.
        if 0 <= pred_f <= 24:
            counts.pred_fret_sum += pred_f
            counts.pred_fret_count += 1
            counts.fret_hist[pred_f] += 1
        if pred_s == gt.s:
            counts.pred_string_correct += 1
        if pred_f == gt.f:
            counts.pred_fret_correct += 1

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


def compute_rhythm_metrics(
    pred_tabs: list[tuple[int, int]],
    pred_tokens: list[str],
    gt_notes: list[Note],
    vocab: Vocab,
    tolerance_ticks: int = 120,
) -> RhythmCounts:
    """
    Compute rhythm metrics by decoding the model's TIME_SHIFT tokens to onsets
    and comparing to ground-truth onsets.

    Uses greedy nearest-onset matching within tolerance_ticks (default 120 = 2 grids).
    """
    from tablature.stream import decode_tab_stream

    # Decode the full token stream (including TIME_SHIFT) to timed notes.
    decoded = decode_tab_stream(pred_tokens)

    # Build predicted timed notes from decoded stream.
    pred_notes = []
    for d in decoded:
        s, f = d["s"], d["f"]
        pitch = pitch_of(s, f, STANDARD)
        pred_notes.append(Note(t=d["t"], dur=d["dur"], pitch=pitch, s=s, f=f))

    # Sort both by onset, then string.
    pred_notes.sort(key=lambda n: (n.t, n.s))
    gt_notes = sorted(gt_notes, key=lambda n: (n.t, n.s))

    # Greedy nearest-onset matching.
    used_pred = set()
    used_gt = set()
    onset_errors = []
    ioi_errors = []
    dur_errors = []

    for gi, gt in enumerate(gt_notes):
        best_pi = None
        best_dt = tolerance_ticks + 1
        for pi, pn in enumerate(pred_notes):
            if pi in used_pred:
                continue
            dt = abs(pn.t - gt.t)
            if dt < best_dt:
                best_dt = dt
                best_pi = pi
        if best_pi is not None and best_dt <= tolerance_ticks:
            used_pred.add(best_pi)
            used_gt.add(gi)
            onset_errors.append(best_dt)
            pn = pred_notes[best_pi]
            dur_errors.append(abs(pn.dur - gt.dur))

            # IOI error: compare interval to previous matched note.
            prev_matched_gt = None
            for gj in range(gi - 1, -1, -1):
                if gj in used_gt:
                    prev_matched_gt = gj
                    break
            prev_matched_pred = None
            for pj in range(best_pi - 1, -1, -1):
                if pj in used_pred:
                    prev_matched_pred = pj
                    break
            if prev_matched_gt is not None and prev_matched_pred is not None:
                gt_ioi = gt.t - gt_notes[prev_matched_gt].t
                pred_ioi = pred_notes[best_pi].t - pred_notes[prev_matched_pred].t
                ioi_errors.append(abs(pred_ioi - gt_ioi))

    matched = len(used_gt)
    missing = len(gt_notes) - matched
    extra = len(pred_notes) - matched

    return RhythmCounts(
        matched=matched,
        missing=missing,
        extra=extra,
        onset_errors=onset_errors,
        ioi_errors=ioi_errors,
        dur_errors=dur_errors,
    )


def aggregate_rhythm(all_counts: list[RhythmCounts]) -> RhythmCounts:
    total = RhythmCounts()
    for c in all_counts:
        total.matched += c.matched
        total.missing += c.missing
        total.extra += c.extra
        total.onset_errors.extend(c.onset_errors)
        total.ioi_errors.extend(c.ioi_errors)
        total.dur_errors.extend(c.dur_errors)
    return total


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

    for _, group in groupby(zip(pred_tabs, gt_notes), key=lambda x: x[1].t):
        chord = list(group)
        if len(chord) > 4:
            simult_gt_4 += 1

        strings = [c[0][0] for c in chord]
        frets = [c[0][1] for c in chord]

        if max(frets) - min(frets) > 5:
            chord_span_gt_5 += 1

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
            covered_notes=total.covered_notes + c.covered_notes,
            covered_tab_correct=total.covered_tab_correct + c.covered_tab_correct,
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
            pred_fret_sum=total.pred_fret_sum + c.pred_fret_sum,
            pred_fret_count=total.pred_fret_count + c.pred_fret_count,
            pred_string_correct=total.pred_string_correct + c.pred_string_correct,
            pred_fret_correct=total.pred_fret_correct + c.pred_fret_correct,
            fret_hist=[
                a + b for a, b in zip(total.fret_hist, c.fret_hist)
            ],
        )
    return total


def print_metrics(counts: EvalCounts, prefix: str = "") -> None:
    """Pretty print metrics."""
    print(f"{prefix}Pitch Accuracy:   {counts.pitch_accuracy():.4f}")
    print(f"{prefix}Tab Accuracy:     {counts.tab_accuracy():.4f}")
    print(f"{prefix}Coverage:         {counts.coverage():.4f} ({counts.covered_notes}/{counts.total_notes})")
    print(f"{prefix}Tab@Pred:         {counts.tab_accuracy_at_pred():.4f} ({counts.covered_tab_correct}/{counts.covered_notes})")
    print(f"{prefix}String Acc:       {counts.string_accuracy():.4f}")
    print(f"{prefix}Fret Acc:         {counts.fret_accuracy():.4f}")
    print(f"{prefix}Mean Fret:        {counts.mean_fret():.2f}")
    print(f"{prefix}High Fret >={HIGH_FRET}:   {counts.high_fret_rate():.4f} ({counts.high_fret_count()}/{counts.pred_fret_count})")
    print(f"{prefix}Validity:         {counts.validity():.4f}")
    print(f"{prefix}Difficulty:       {counts.difficulty():.4f}")
    print(f"{prefix}Boundary Acc:     {counts.boundary_accuracy():.4f} ({counts.boundary_correct}/{counts.boundary_total})")
    print(f"{prefix}Interior Acc:     {counts.interior_accuracy():.4f} ({counts.interior_correct}/{counts.interior_total})")
    print(f"{prefix}Error Decomp:     pitch={counts.pitch_errors} position={counts.position_errors}")
    print(f"{prefix}Playability:      overlap={counts.same_string_overlap} span>5={counts.chord_span_gt_5} simult>4={counts.simult_gt_4}")
    print(f"{prefix}Total Notes:      {counts.total_notes}")


def print_rhythm_metrics(counts: RhythmCounts, prefix: str = "") -> None:
    """Pretty print rhythm metrics."""
    if counts.matched == 0:
        print(f"{prefix}Rhythm: no matched notes")
        return
    print(f"{prefix}Matched/Missing/Extra: {counts.matched}/{counts.missing}/{counts.extra}")
    print(f"{prefix}Onset Hit @20/60/120: {counts.onset_hit_rate(20):.4f}/{counts.onset_hit_rate(60):.4f}/{counts.onset_hit_rate(120):.4f}")
    print(f"{prefix}Median Onset Error:   {counts.median_onset_error():.1f} ticks")
    print(f"{prefix}Mean Onset Error:     {counts.mean_onset_error():.1f} ticks")
    print(f"{prefix}Median IOI Error:     {counts.median_ioi_error():.1f} ticks")
    print(f"{prefix}Mean IOI Error:       {counts.mean_ioi_error():.1f} ticks")
    print(f"{prefix}Median Dur Error:     {counts.median_dur_error():.1f} ticks")
    print(f"{prefix}Mean Dur Error:       {counts.mean_dur_error():.1f} ticks")