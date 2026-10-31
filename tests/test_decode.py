"""Plain-assert tests for the pitch/low-position decoding mask."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.decode import PitchValidLogitsProcessorV2  # noqa: E402
from tablature.tokenizer import (  # noqa: E402
    DEFAULT_VOCAB,
    EOS_TOKEN,
    note_on_token,
    tab_token,
)

FAILS: list[str] = []
VOCAB = DEFAULT_VOCAB


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def _scores(pitches: list[int], **kwargs) -> torch.Tensor:
    enc = torch.tensor(
        [[VOCAB[note_on_token(p)] for p in pitches]], dtype=torch.long
    )
    proc = PitchValidLogitsProcessorV2(VOCAB, enc, **kwargs)
    scores = torch.zeros(1, VOCAB.size, dtype=torch.float32)
    prompt = torch.tensor([[VOCAB[EOS_TOKEN]]], dtype=torch.long)
    proc(prompt, scores)
    return scores[0]


def tid(s: int, f: int) -> int:
    return VOCAB[tab_token(s, f)]


def test_low_position_mask() -> None:
    # E4 can be played at fret 0 (high E), 5 (B), 9 (G), 14 (D), 19 (A), 24 (low E).
    scores = _scores([64], max_fret=12)
    check("open high-E position allowed", scores[tid(6, 0)].item() == 0.0)
    check("G-string fret 9 allowed", scores[tid(4, 9)].item() == 0.0)
    check("D-string fret 14 masked", scores[tid(3, 14)].item() == -1e4)
    check("A-string fret 19 masked", scores[tid(2, 19)].item() == -1e4)
    check("low-E fret 24 masked", scores[tid(1, 24)].item() == -1e4)
    check("unrelated pitch token masked", scores[tid(6, 24)].item() == -1e4)
    check("non-TAB tokens untouched", scores[VOCAB[EOS_TOKEN]].item() == 0.0)


def test_high_note_fallback() -> None:
    # E6 only exists at fret 24 on the high E; the low-position rule must not
    # make the note unplayable.
    scores = _scores([88], max_fret=12)
    check("E6 fret 24 allowed when no low option", scores[tid(6, 24)].item() == 0.0)
    check("invalid E6 position masked", scores[tid(5, 19)].item() == -1e4)


def test_fret_penalty() -> None:
    scores = _scores([64], max_fret=12, fret_penalty=0.1)
    check("open string has no penalty", abs(scores[tid(6, 0)].item()) < 1e-6)
    check(
        "B-string fret 5 penalised by 0.5",
        abs(scores[tid(5, 5)].item() + 0.5) < 1e-6,
        str(scores[tid(5, 5)].item()),
    )
    check(
        "G-string fret 9 penalised by 0.9",
        abs(scores[tid(4, 9)].item() + 0.9) < 1e-6,
        str(scores[tid(4, 9)].item()),
    )


def test_optional_mask() -> None:
    scores = _scores([64], max_fret=None)
    check("high position allowed without max_fret", scores[tid(3, 14)].item() == 0.0)
    check("invalid position still masked", scores[tid(6, 24)].item() == -1e4)


def test_validation() -> None:
    enc = torch.tensor([[VOCAB[note_on_token(64)]]], dtype=torch.long)
    try:
        PitchValidLogitsProcessorV2(VOCAB, enc, max_fret=25)
    except ValueError:
        check("max_fret above fretboard rejected", True)
    else:
        check("max_fret above fretboard rejected", False)
    try:
        PitchValidLogitsProcessorV2(VOCAB, enc, fret_penalty=-0.1)
    except ValueError:
        check("negative fret_penalty rejected", True)
    else:
        check("negative fret_penalty rejected", False)


def main() -> int:
    test_low_position_mask()
    test_high_note_fallback()
    test_fret_penalty()
    test_optional_mask()
    test_validation()
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
