"""Plain-assert tests for the pre-tokenised chunk store.

Run with ``python tests/test_chunk_store.py``. Builds a tiny synthetic shard in
a temporary directory, so it does not need the real processed corpus.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.fretboard import STANDARD  # noqa: E402
from core.schema import Note  # noqa: E402
from corpus.chunk_store import (  # noqa: E402
    ChunkStore,
    build_chunk_store,
    get_chunk_store,
)
from corpus.dataset import ShardWriter, iter_tracks, notes_to_array, records_to_notes  # noqa: E402
from model.chunk import ChunkConfig, make_training_chunks  # noqa: E402
from tablature.tokenizer import DEFAULT_VOCAB  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} {detail}")
        print(f"FAIL {name} {detail}")
    else:
        print(f"ok   {name}")


def make_track(n: int, start: int, seed: int) -> list[Note]:
    rng = np.random.default_rng(seed)
    notes = []
    t = start
    for _ in range(n):
        s = int(rng.integers(1, 7))
        f = int(rng.integers(0, 12))
        notes.append(Note(t=t, dur=240, pitch=STANDARD[s - 1] + f, s=s, f=f))
        t += 240 if rng.random() < 0.6 else 480
    return notes


def write_corpus(root: Path) -> None:
    writer = ShardWriter(root)
    for i in range(6):
        split = "train" if i < 4 else "valid"
        notes = make_track(40 + 5 * i, start=20, seed=i)
        writer.add(
            {"id": i, "split": split, "source": f"t{i}.mid", "song": f"s{i}"},
            notes_to_array(notes),
        )
    writer.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_corpus(tmp_path)
    return tmp_path


def reference_chunks(root: Path, split: str) -> list:
    out = []
    for meta, arr in iter_tracks(root, split=split):
        for chunk in make_training_chunks(
            records_to_notes(arr), DEFAULT_VOCAB, meta["id"], ChunkConfig(max_tokens=512)
        ):
            if chunk.enc_tokens and chunk.labels:
                out.append(chunk)
    return out


def test_matches_reference(root: Path) -> None:
    store = build_chunk_store(root, "train", DEFAULT_VOCAB, 512)
    ref = reference_chunks(root, "train")
    check("chunk count matches direct chunking", len(store) == len(ref),
          f"{len(store)} vs {len(ref)}")
    ok = True
    for i, chunk in enumerate(ref):
        enc = store.enc[store.enc_off[i]:store.enc_off[i + 1]]
        dec = store.dec[store.dec_off[i]:store.dec_off[i + 1]]
        lab = store.lab[store.lab_off[i]:store.lab_off[i + 1]]
        if (enc.tolist() != chunk.enc_tokens
                or dec.tolist() != chunk.dec_tokens
                or lab.tolist() != chunk.labels):
            ok = False
            break
    check("tokens identical to direct chunking", ok)
    check("split filter respected", len(build_chunk_store(root, "valid", DEFAULT_VOCAB, 512)) > 0)
    check("track ids recorded", len(store.track_ids) == len(store))


def test_round_trip(root: Path) -> None:
    store = build_chunk_store(root, "train", DEFAULT_VOCAB, 512)
    path = root / "store.npz"
    store.save(path)
    back = ChunkStore.load(path)
    check("disk round-trip preserves arrays",
          np.array_equal(back.enc, store.enc)
          and np.array_equal(back.dec, store.dec)
          and np.array_equal(back.lab, store.lab)
          and np.array_equal(back.lab_off, store.lab_off)
          and np.array_equal(back.track_ids, store.track_ids))


def test_cache(root: Path) -> None:
    first = get_chunk_store(root, "train", DEFAULT_VOCAB, 512)
    cache_dir = root / ".chunk_cache"
    files = sorted(cache_dir.glob("*.npz"))
    check("cache file written", len(files) == 1, str(files))
    second = get_chunk_store(root, "train", DEFAULT_VOCAB, 512)
    check("cache hit returns same tokens",
          np.array_equal(first.enc, second.enc)
          and np.array_equal(first.lab_off, second.lab_off))

    # A different token cap must not reuse the 512-token cache.
    other = get_chunk_store(root, "train", DEFAULT_VOCAB, 256)
    check("token cap changes the cache key", len(list(cache_dir.glob("*.npz"))) == 2)
    check("smaller cap yields shorter chunks",
          other.enc_off[-1] <= first.enc_off[-1])

    # Touching a shard file invalidates the cache (content fingerprint).
    shard = sorted(p for p in root.glob("shard-*") if p.is_dir())[0]
    target = shard / "tracks.jsonl"
    import os
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1))
    get_chunk_store(root, "train", DEFAULT_VOCAB, 512)
    check("shard change invalidates the cache",
          len(list(cache_dir.glob("*.npz"))) == 3)

    # Read-only cache directory must fall back to memory, not crash.
    blocked = root / "readonly"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        store = get_chunk_store(
            root, "train", DEFAULT_VOCAB, 512, cache_dir=blocked
        )
        check("unwritable cache falls back to memory", len(store) > 0)
    finally:
        blocked.chmod(0o700)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "corpus"
        write_corpus(root)
        test_matches_reference(root)
        test_round_trip(root)
        test_cache(root)
    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
