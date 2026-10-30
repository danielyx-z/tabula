import sys
sys.path.insert(0, "src")
from fretwire.dataset import iter_tracks
from fretwire.schema import Note
from fretwire.tokenizer import Vocab, PAD_TOKEN, EOS_TOKEN
from fretwire.metrics import decode_predictions
from fretwire.chunk import ChunkConfig, make_inference_chunks
from fretwire.decode import constrained_beam_search, greedy_decode
from fretwire.fretboard import pitch_of, STANDARD
import torch
from transformers import T5ForConditionalGeneration

vocab = Vocab.load("vocab_v1_full.json")
model = T5ForConditionalGeneration.from_pretrained("runs/pretrain_t5_v5/best")
model.to("cuda").eval()
model.to(torch.bfloat16)

gt_notes = []
for meta, notes_arr in iter_tracks("data/processed", split="train"):
    if int(meta.get("id", -1)) == 6:
        gt_notes = [Note(t=int(r["t"]), dur=int(r["dur"]), pitch=int(r["pitch"]), s=int(r["s"]), f=int(r["f"])) for r in notes_arr]
        break

gt_notes.sort(key=lambda n: (n.t, n.pitch, n.s, n.f))

chunk_cfg = ChunkConfig(max_tokens=512)
chunks = make_inference_chunks(gt_notes, vocab, 6, chunk_cfg)

pad_id = vocab[PAD_TOKEN]
eos_id = vocab[EOS_TOKEN]

all_pred_tabs = []
for chunk_idx, chunk in enumerate(chunks):
    enc = torch.tensor([chunk.enc_tokens], dtype=torch.long, device="cuda")
    attn = (enc != pad_id).long()
    prompt = torch.tensor([chunk.dec_tokens], dtype=torch.long, device="cuda")
    max_new = min(512, max(16, 6 * chunk.meta["num_notes"] + 16))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        gen = constrained_beam_search(
            model, enc, vocab, num_beams=4, max_length=512,
            use_pitch_constraint=True, decoder_input_ids=prompt,
            attention_mask=attn, max_new_tokens=max_new,
            prompt_tabs=0,
        )
    gen_ids = gen[0, prompt.size(1):].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]
    pred_tabs = decode_predictions(gen_ids, vocab)
    all_pred_tabs.extend(pred_tabs)
    print(f"Chunk {chunk_idx+1}: {len(pred_tabs)} tabs")

print(f"\nGT notes: {len(gt_notes)}, Pred tabs: {len(all_pred_tabs)}")
print(f"\nFirst 15 note-by-note comparison:")
print(f"{'#':>3} {'GT(s,f)':>9} {'GT pitch':>7} {'Pred(s,f)':>10} {'Pred pitch':>9} {'P':>6} {'T':>6}")
pitch_matches = 0
tab_matches = 0
for i in range(min(len(gt_notes), len(all_pred_tabs))):
    gt = gt_notes[i]
    pred = all_pred_tabs[i] if i < len(all_pred_tabs) else None
    gt_pitch = pitch_of(gt.s, gt.f, STANDARD)
    if pred:
        pred_pitch = pitch_of(pred[0], pred[1], STANDARD)
        p_match = gt_pitch == pred_pitch
        t_match = (gt.s, gt.f) == pred
        pitch_matches += p_match
        tab_matches += t_match
        print(f"  {i:3} ({gt.s},{gt.f})   {gt_pitch:>7}   ({pred[0]},{pred[1]})   {pred_pitch:>9}   {'✓' if p_match else '✗'}   {'✓' if t_match else '✗'}")
    else:
        print(f"  {i:3} ({gt.s},{gt.f})   {gt_pitch:>7}   {'NONE':>10}   {'N/A':>9}   {'✗'}   {'✗'}")

print(f"\nPitch matches: {pitch_matches}/{min(len(gt_notes), len(all_pred_tabs))}")
print(f"Tab matches: {tab_matches}/{min(len(gt_notes), len(all_pred_tabs))}")
