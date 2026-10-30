from pathlib import Path

# Fix parse/midi.py - scale ticks from file resolution to PPQ=960
p = Path("src/fretwire/parse/midi.py")
s = p.read_text()

old = """def _ticks_from_seconds(pm: pretty_midi.PrettyMIDI, seconds: float) -> int:
    \"\"\"Convert seconds to ticks at PPQ, following the tempo map.\"\"\"
    # pretty_midi's get_tick() does exactly this
    return pm.time_to_tick(seconds)"""
new = """def _ticks_from_seconds(pm: pretty_midi.PrettyMIDI, seconds: float) -> int:
    \"\"\"Convert seconds to ticks at PPQ 960, following the tempo map.

    pretty_midi's time_to_tick() returns ticks at the file's own resolution
    (e.g. 220 for basic-pitch output).  Scale to our standard PPQ=960 so
    TIME_SHIFT tokens match the training distribution.
    \"\"\"
    ticks_at_file_res = pm.time_to_tick(seconds)
    return int(round(ticks_at_file_res * PPQ / pm.resolution))"""
assert s.count(old) == 1, "parse/midi.py: old block not found"
s = s.replace(old, new)

p.write_text(s)
print("src/fretwire/parse/midi.py updated")

# Fix cli.py - pitch constraint, beam default, prompt_tabs
p = Path("src/fretwire/cli.py")
s = p.read_text()

old = """            if args.beam > 1:
                gen = constrained_beam_search(
                    model, enc, vocab, num_beams=args.beam, max_length=args.max_len,
                    use_pitch_constraint=False, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                )
            else:
                gen = greedy_decode(
                    model, enc, vocab, max_length=args.max_len,
                    use_pitch_constraint=False, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                )"""
new = """            prompt_tabs = 1 if chunk.meta.get("has_context") else 0
            if args.beam > 1:
                gen = constrained_beam_search(
                    model, enc, vocab, num_beams=args.beam, max_length=args.max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )
            else:
                gen = greedy_decode(
                    model, enc, vocab, max_length=args.max_len,
                    use_pitch_constraint=True, decoder_input_ids=prompt,
                    attention_mask=attn, max_new_tokens=max_new,
                    prompt_tabs=prompt_tabs,
                )"""
assert s.count(old) == 1, "cli.py: decode block not found"
s = s.replace(old, new)

old = 'ap.add_argument("--beam", type=int, default=1, help="Beam size (1=greedy)")'
new = 'ap.add_argument("--beam", type=int, default=4, help="Beam size (1=greedy)")'
assert s.count(old) == 1, "cli.py: beam arg not found"
s = s.replace(old, new)

p.write_text(s)
print("src/fretwire/cli.py updated")

PYEOF
