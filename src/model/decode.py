"""Decoding helpers for the T5 model.

The paper decodes normally and repairs pitch in post-processing, so the
pitch-valid processor is optional and disabled by default.  T5 supports beam
search natively, so ``constrained_beam_search`` no longer falls back to greedy.

The processor can also apply a low-position prior: when a source pitch has at
least one playable position at or below ``max_fret``, TAB tokens above that
fret are masked.  This removes gratuitous high-position fingerings while still
allowing genuinely high notes (e.g. E6 has no position below fret 24).
``fret_penalty`` adds a soft ``-lambda * fret`` bias among the allowed
positions, which breaks ties in favour of lower positions.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from core.fretboard import STANDARD, positions
from tablature.tokenizer import EOS_TOKEN, PAD_TOKEN, Vocab, parse_note_on, tab_token


class PitchValidLogitsProcessorV2(LogitsProcessor):
    """Mask TAB tokens that are not playable for the current source pitch.

    ``prompt_tabs`` is the number of TAB tokens already present in a decoder
    prompt (for example the previous chunk's context note).  Those tokens must
    not consume the next source pitch.

    ``max_fret`` restricts each note to the lowest position region when a
    playable position exists at or below that fret.  ``fret_penalty`` biases
    the remaining valid positions towards lower frets; both are inference-only
    and do not change the training loss.
    """

    def __init__(
        self,
        vocab: Vocab,
        encoder_input_ids: torch.LongTensor,
        tuning: tuple[int, ...] = STANDARD,
        frets: int = 24,
        prompt_tabs: int = 0,
        max_fret: int | None = None,
        fret_penalty: float = 0.0,
    ):
        if max_fret is not None and not (0 <= max_fret <= frets):
            raise ValueError(
                f"max_fret must be between 0 and {frets}, got {max_fret}"
            )
        if fret_penalty < 0:
            raise ValueError(f"fret_penalty must be >= 0, got {fret_penalty}")

        if encoder_input_ids.dim() == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.vocab = vocab
        self.encoder_input_ids = encoder_input_ids
        self.tuning = tuning
        self.frets = frets
        self.prompt_tabs = prompt_tabs
        self.max_fret = max_fret
        self.fret_penalty = float(fret_penalty)

        self.tab_token_ids: set[int] = {
            i for i, tok in enumerate(vocab.tokens) if tok.startswith("TAB<")
        }
        self.tab_fret: dict[int, int] = {}
        for s in range(1, 7):
            for f in range(0, frets + 1):
                tok = tab_token(s, f)
                if tok in vocab.token_to_id:
                    self.tab_fret[vocab.token_to_id[tok]] = f
        device = encoder_input_ids.device

        self.valid_tabs_for_pitch: dict[int, set[int]] = {}
        self.masked_tab_ids_for_pitch: dict[int, torch.Tensor] = {}
        self.penalty_bias_for_pitch: dict[int, torch.Tensor] | None = (
            {} if self.fret_penalty > 0 else None
        )
        for pitch in range(128):
            valid: set[int] = set()
            for s, f in positions(pitch, tuning, frets):
                tok = tab_token(s, f)
                if tok in vocab.token_to_id:
                    valid.add(vocab.token_to_id[tok])
            self.valid_tabs_for_pitch[pitch] = valid

            if max_fret is not None:
                low = {tid for tid in valid if self.tab_fret[tid] <= max_fret}
                allowed = low if low else valid
            else:
                allowed = valid
            self.masked_tab_ids_for_pitch[pitch] = torch.tensor(
                sorted(self.tab_token_ids - allowed),
                dtype=torch.long,
                device=device,
            )

            if self.penalty_bias_for_pitch is not None:
                bias = torch.zeros(vocab.size, dtype=torch.float32, device=device)
                for tid in allowed:
                    bias[tid] = -self.fret_penalty * self.tab_fret[tid]
                self.penalty_bias_for_pitch[pitch] = bias

        self.all_tab_ids = torch.tensor(
            sorted(self.tab_token_ids), dtype=torch.long, device=device
        )

        self.shift_token_ids: set[int] = {
            i for i, tok in enumerate(vocab.tokens) if tok.startswith("TIME_SHIFT<")
        }
        # tab token id -> string number (1..6)
        self.tab_string: dict[int, int] = {}
        for s in range(1, 7):
            for f in range(0, frets + 1):
                tok = tab_token(s, f)
                if tok in vocab.token_to_id:
                    self.tab_string[vocab.token_to_id[tok]] = s

        note_on_ids = {
            i for i, tok in enumerate(vocab.tokens) if tok.startswith("NOTE_ON<")
        }
        self.pitches_per_row: list[list[int]] = []
        for b in range(encoder_input_ids.size(0)):
            pitches = [
                parse_note_on(vocab.id_to_token[tok])
                for tok in encoder_input_ids[b].tolist()
                if tok in note_on_ids
            ]
            self.pitches_per_row.append(pitches)

    def _current_pitch(self, row: int, generated: torch.Tensor) -> int | None:
        pitches = self.pitches_per_row[row % max(1, len(self.pitches_per_row))]
        tab_count = sum(
            1 for tok in generated if tok in self.tab_token_ids
        ) - self.prompt_tabs
        if 0 <= tab_count < len(pitches):
            return pitches[tab_count]
        return None

    def _current_onset_strings(self, generated: list[int]) -> set[int]:
        """Strings already used at the current onset.

        The decoder stream emits TAB tokens back-to-back for a chord, then a
        TIME_SHIFT to advance to the next event time, so the strings used at
        the current onset are the TAB tokens since the most recent TIME_SHIFT.
        Masking these prevents two simultaneous notes being assigned to one
        string (which post-processing would otherwise have to drop).
        """
        strings: set[int] = set()
        for tok in reversed(generated):
            if tok in self.shift_token_ids:
                break
            s = self.tab_string.get(tok)
            if s is not None:
                strings.add(s)
        return strings

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for b in range(input_ids.size(0)):
            pitch = self._current_pitch(b, input_ids[b].tolist())
            if pitch is None:
                scores[b, self.all_tab_ids] = -1e4
                continue
            scores[b, self.masked_tab_ids_for_pitch[pitch]] = -1e4
            # Onset-aware string masking: one note per string per onset.
            allowed = self.valid_tabs_for_pitch.get(pitch, set())
            used = self._current_onset_strings(input_ids[b].tolist())
            if used:
                to_mask = [tid for tid in allowed if self.tab_string.get(tid) in used]
                # Only apply if it does not wipe out every pitch-valid position;
                # a same-onset string collision is better left to post-processing
                # than forcing a wrong/invalid pitch.
                if to_mask and (allowed - set(to_mask)):
                    for tid in to_mask:
                        scores[b, tid] = -1e4
            if self.penalty_bias_for_pitch is not None:
                scores[b] = scores[b] + self.penalty_bias_for_pitch[pitch]
        return scores


# Backwards-compatible alias.
PitchValidLogitsProcessor = PitchValidLogitsProcessorV2


def make_logits_processor(
    vocab: Vocab,
    encoder_input_ids: torch.LongTensor,
    tuning: tuple[int, ...] = STANDARD,
    prompt_tabs: int = 0,
    max_fret: int | None = None,
    fret_penalty: float = 0.0,
) -> PitchValidLogitsProcessorV2:
    return PitchValidLogitsProcessorV2(
        vocab,
        encoder_input_ids,
        tuning,
        prompt_tabs=prompt_tabs,
        max_fret=max_fret,
        fret_penalty=fret_penalty,
    )


def _generate_kwargs(vocab: Vocab) -> dict:
    return {
        "eos_token_id": vocab[EOS_TOKEN],
        "pad_token_id": vocab[PAD_TOKEN],
        "do_sample": False,
    }


def greedy_decode(
    model,
    encoder_input_ids: torch.LongTensor,
    vocab: Vocab,
    max_length: int = 512,
    tuning: tuple[int, ...] = STANDARD,
    min_length: int = 0,
    use_pitch_constraint: bool = False,
    decoder_input_ids: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    prompt_tabs: int = 0,
    max_new_tokens: int | None = None,
    max_fret: int | None = None,
    fret_penalty: float = 0.0,
) -> torch.LongTensor:
    """Greedy decoding; the paper's default is unconstrained."""
    kwargs = _generate_kwargs(vocab)
    kwargs["num_beams"] = 1
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    else:
        kwargs["max_length"] = max_length
    if min_length:
        kwargs["min_length"] = min_length
    if decoder_input_ids is not None:
        kwargs["decoder_input_ids"] = decoder_input_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if use_pitch_constraint:
        kwargs["logits_processor"] = [
            make_logits_processor(
                vocab,
                encoder_input_ids,
                tuning,
                prompt_tabs=prompt_tabs,
                max_fret=max_fret,
                fret_penalty=fret_penalty,
            )
        ]
    return model.generate(encoder_input_ids, **kwargs)


def constrained_beam_search(
    model,
    encoder_input_ids: torch.LongTensor,
    vocab: Vocab,
    num_beams: int = 4,
    max_length: int = 512,
    tuning: tuple[int, ...] = STANDARD,
    use_pitch_constraint: bool = True,
    decoder_input_ids: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    prompt_tabs: int = 0,
    max_new_tokens: int | None = None,
    max_fret: int | None = None,
    fret_penalty: float = 0.0,
) -> torch.LongTensor:
    """Beam search with an optional pitch-valid and low-position mask.

    ``use_pitch_constraint`` defaults to True for backwards compatibility, but
    the paper-faithful evaluation path uses plain T5 generation plus
    post-processing.  ``max_fret`` and ``fret_penalty`` are applied only when
    the pitch constraint is enabled.
    """
    kwargs = _generate_kwargs(vocab)
    kwargs["num_beams"] = num_beams
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    else:
        kwargs["max_length"] = max_length
    if decoder_input_ids is not None:
        kwargs["decoder_input_ids"] = decoder_input_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if use_pitch_constraint:
        kwargs["logits_processor"] = [
            make_logits_processor(
                vocab,
                encoder_input_ids,
                tuning,
                prompt_tabs=prompt_tabs,
                max_fret=max_fret,
                fret_penalty=fret_penalty,
            )
        ]
    return model.generate(encoder_input_ids, **kwargs)
