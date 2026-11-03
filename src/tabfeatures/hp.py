"""Conservative, audio-gated hammer-on / pull-off inference.

The model-only rules remain deliberately strict: adjacent notes on one string,
a short gap, a small interval, no open strings, no chords, and the first note
still ringing according to the tab duration.

When audio evidence is available, the decision is gated further.  A real H/P
requires the first note to still be present at the second onset, and it should
not have a full separate pluck attack.  A strong second attack, a decayed first
note, or a second note with no pitched support rejects the candidate.

Candidates are written to ``hp_review.json`` either way.  Only accepted
candidates are returned in ``features``.  Manual overlays remain authoritative.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

from .evidence import (
    AudioAnalysis,
    best_event_for_note,
    event_audio_class,
    local_median_strength,
    non_note_score,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def infer_hp(
    notes: list[dict[str, Any]],
    *,
    max_gap_ticks: int = 480,
    max_interval: int = 7,
    ring_tolerance: int = 60,
    skip_chords: bool = True,
    grid_ticks: int = 240,
    events: list[dict[str, Any]] | None = None,
    analysis: AudioAnalysis | None = None,
    offset_sec: float = 0.0,
    tick_seconds: Callable[[int], float] | None = None,
    min_confidence: float = 0.0,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    """Return ``{(tick, string): feature}`` and a review list.

    Without ``events`` / ``analysis`` this preserves the original model-only
    behavior, which is useful for unit tests and offline use.  The stitching
    pipeline passes audio evidence and a confidence threshold.
    """
    if max_gap_ticks <= 0:
        return {}, []

    active = [note for note in notes if not note.get("_drop")]
    onset_counts = Counter(int(note["t"]) for note in active)

    by_string: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in active:
        by_string[int(note["s"])].append(note)
    for string in by_string:
        by_string[string].sort(key=lambda n: (int(n["t"]), int(n["f"])))

    features: dict[tuple[int, int], dict[str, Any]] = {}
    review: list[dict[str, Any]] = []
    audio_checked = events is not None

    for string, sequence in by_string.items():
        for first, second in zip(sequence, sequence[1:]):
            t1 = int(first["t"])
            t2 = int(second["t"])
            gap = t2 - t1
            if gap <= 0 or gap > int(max_gap_ticks):
                continue
            if int(grid_ticks) > 0 and int(round(t1 / grid_ticks)) == int(
                round(t2 / grid_ticks)
            ):
                # Both notes would render in the same 1/16 cell.
                continue
            if t2 > t1 + int(first.get("dur", 0)) + int(ring_tolerance):
                continue

            interval = abs(int(second["pitch"]) - int(first["pitch"]))
            if interval < 1 or interval > int(max_interval):
                continue
            if int(first.get("f", 0)) <= 0 or int(second.get("f", 0)) <= 0:
                continue
            if skip_chords and (
                onset_counts[t1] > 1 or onset_counts[t2] > 1
            ):
                continue
            if (t1, string) in features or (t2, string) in features:
                continue

            kind = "h" if int(second["pitch"]) > int(first["pitch"]) else "p"
            timing_confidence = 0.5 + 0.5 * (1.0 - gap / float(max_gap_ticks))
            if interval > 5:
                timing_confidence *= 0.8
            timing_confidence = _clamp(timing_confidence)

            first_event = (
                best_event_for_note(first, events, grid_ticks=grid_ticks)
                if events is not None
                else None
            )
            second_event = (
                best_event_for_note(second, events, grid_ticks=grid_ticks)
                if events is not None
                else None
            )

            audio_score: float | None = None
            rejection: str | None = None
            details: dict[str, Any] = {
                "first_event": None,
                "second_event": None,
                "attack_ratio": None,
                "first_sustain": None,
                "second_pitch_support": None,
                "audio_score": None,
            }

            if first_event is not None:
                details["first_event"] = {
                    "time": round(float(first_event.get("time", 0.0)), 6),
                    "class": event_audio_class(first_event),
                    "strength": round(float(first_event.get("strength", 0.0) or 0.0), 6),
                }
            if second_event is not None:
                details["second_event"] = {
                    "time": round(float(second_event.get("time", 0.0)), 6),
                    "class": event_audio_class(second_event),
                    "strength": round(float(second_event.get("strength", 0.0) or 0.0), 6),
                }

            if events is not None:
                # The second event, when present, is the closest audio onset to
                # the second note.  If the detector missed it, use the MIDI
                # tick converted to audio time; that absence is weak legato
                # evidence rather than a failure by itself.
                second_time = None
                if second_event is not None:
                    second_time = float(second_event.get("time", 0.0))
                elif tick_seconds is not None:
                    second_time = float(tick_seconds(t2)) - float(offset_sec)

                first_time = None
                if first_event is not None:
                    first_time = float(first_event.get("time", 0.0))
                elif tick_seconds is not None:
                    first_time = float(tick_seconds(t1)) - float(offset_sec)

                attack_ratio = 0.0
                if second_event is not None:
                    local_median = local_median_strength(
                        events,
                        around_time=float(second_event.get("time", 0.0)),
                    )
                    first_strength = (
                        float(first_event.get("strength", 0.0) or 0.0)
                        if first_event is not None
                        else 0.0
                    )
                    second_strength = float(second_event.get("strength", 0.0) or 0.0)
                    attack_ratio = second_strength / max(
                        first_strength, local_median, 1e-6
                    )
                    details["attack_ratio"] = round(attack_ratio, 6)

                if analysis is not None and first_time is not None and second_time is not None:
                    first_support = analysis.pitch_support(
                        first_time,
                        int(first["pitch"]),
                        start=0.01,
                        end=0.20,
                    )
                    sustain = analysis.pitch_support(
                        second_time,
                        int(first["pitch"]),
                        start=-0.08,
                        end=0.02,
                    )
                    second_support = analysis.pitch_support(
                        second_time,
                        int(second["pitch"]),
                        start=0.01,
                        end=0.18,
                    )
                    sustain_ratio = sustain / max(first_support, 1e-6)
                    details["first_sustain"] = round(sustain_ratio, 6)
                    details["second_pitch_support"] = round(second_support, 6)

                    attack_score = 1.0 - _clamp(attack_ratio / 0.75)
                    sustain_score = _clamp(sustain_ratio / 0.35)
                    pitch_score = _clamp(second_support / 0.12)
                    audio_score = (
                        0.45 * attack_score
                        + 0.35 * sustain_score
                        + 0.20 * pitch_score
                    )

                    if attack_ratio >= 0.85:
                        rejection = "separate_attack"
                    elif sustain_ratio < 0.20:
                        rejection = "first_note_not_ringing"
                    elif second_support < 0.04:
                        rejection = "weak_second_pitch"
                    elif second_event is not None and non_note_score(
                        second_event, second_support
                    ) >= 0.75:
                        rejection = "second_note_is_non_note"
                elif second_event is not None:
                    # Event-only fallback: a full-strength second attack is
                    # enough to reject, even without the STFT analysis.
                    attack_score = 1.0 - _clamp(attack_ratio / 0.75)
                    audio_score = attack_score
                    if attack_ratio >= 0.85:
                        rejection = "separate_attack"

                details["audio_score"] = (
                    round(audio_score, 6) if audio_score is not None else None
                )

            if audio_score is None:
                confidence = timing_confidence
            else:
                confidence = 0.35 * timing_confidence + 0.65 * audio_score
            confidence = _clamp(confidence)

            if rejection is None and audio_checked and audio_score is not None:
                if confidence < float(min_confidence):
                    rejection = "below_confidence"
            accepted = rejection is None

            item = {
                "from": {
                    "t": t1,
                    "s": string,
                    "f": int(first["f"]),
                    "pitch": int(first["pitch"]),
                },
                "to": {
                    "t": t2,
                    "s": string,
                    "f": int(second["f"]),
                    "pitch": int(second["pitch"]),
                },
                "kind": kind,
                "gap_ticks": gap,
                "interval_semitones": interval,
                "confidence": round(confidence, 6),
                "accepted": accepted,
                "rejection_reason": rejection,
            }
            item.update(details)
            review.append(item)
            if not accepted:
                continue

            features[(t2, string)] = {
                "hammer": True,
                "hp": kind,
                "source": "inferred",
                "confidence": round(confidence, 6),
            }

    return features, review
