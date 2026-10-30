"""Decoding helpers for the T5 model.

The paper decodes normally and repairs pitch in post-processing, so the
pitch-valid processor is optional and disabled by default.  T5 supports beam
search natively, so ``constrained_beam_search`` no longer falls back to greedy.
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
    """

    def __init__(
        self,
        vocab: Vocab,
        encoder_input_ids: torch.LongTensor,
        tuning: tuple[int, ...] = STANDARD,
        frets: int = 24,
        prompt_tabs: int = 0,
    ):
        if encoder_input_ids.dim() == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.vocab = vocab
        self.encoder_input_ids = encoder_input_ids
        self.tuning = tuning
        self.frets = frets
        self.prompt_tabs = prompt_tabs

        self.tab_token_ids: set[int] = {
            i for i, tok in enumerate(vocab.tokens) if tok.startswith("TAB<")
        }
        device = encoder_input_ids.device

        self.valid_tabs_for_pitch: dict[int, set[int]] = {}
        self.invalid_tab_ids_for_pitch: dict[int, torch.Tensor] = {}
        for pitch in range(128):
            valid: set[int] = set()
            for s, f in positions(pitch, tuning, frets):
                tok = tab_token(s, f)
                if tok in vocab.token_to_id:
                    valid.add(vocab.token_to_id[tok])
            self.valid_tabs_for_pitch[pitch] = valid
            self.invalid_tab_ids_for_pitch[pitch] = torch.tensor(
                sorted(self.tab_token_ids - valid), dtype=torch.long, device=device
            )
        self.all_tab_ids = torch.tensor(
            sorted(self.tab_token_ids), dtype=torch.long, device=device
        )

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

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for b in range(input_ids.size(0)):
            pitch = self._current_pitch(b, input_ids[b].tolist())
            invalid = (
                self.all_tab_ids
                if pitch is None
                else self.invalid_tab_ids_for_pitch[pitch]
            )
            scores[b, invalid] = -1e4
        return scores


# Backwards-compatible alias.
PitchValidLogitsProcessor = PitchValidLogitsProcessorV2


def make_logits_processor(
    vocab: Vocab,
    encoder_input_ids: torch.LongTensor,
    tuning: tuple[int, ...] = STANDARD,
    prompt_tabs: int = 0,
) -> PitchValidLogitsProcessorV2:
    return PitchValidLogitsProcessorV2(
        vocab, encoder_input_ids, tuning, prompt_tabs=prompt_tabs
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
) -> torch.LongTensor:
    """Beam search with an optional pitch-valid mask.

    ``use_pitch_constraint`` defaults to True for backwards compatibility, but
    the paper-faithful evaluation path uses plain T5 generation plus
    post-processing.
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
            )
        ]
    return model.generate(encoder_input_ids, **kwargs)
