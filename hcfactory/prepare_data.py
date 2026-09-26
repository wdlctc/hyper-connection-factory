"""Download + tokenize a corpus into flat uint16 shards.

    python -m hcfactory.prepare_data --dataset fineweb-edu --train-tokens 200_000_000
    python -m hcfactory.prepare_data --dataset tinystories --train-tokens 50_000_000

Writes ``data/<dataset>/{train,val}.bin`` with GPT-2 BPE tokens (vocab 50257).

``--source parquet`` downloads the dataset's parquet shards with huggingface_hub
and tokenizes them with a process pool -- the fast path for multi-billion-token
GPU-scale corpora:

    python -m hcfactory.prepare_data --source parquet --train-tokens 8_000_000_000 --workers 64

``--source rows`` streams documents through the Hugging Face datasets-server
REST API instead of the ``datasets`` library (no pyarrow needed; handy for
small laptop-scale corpora or slow networks).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DATASETS = {
    # name: (hf path, hf config, split, text field)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "tinystories": ("roneneldan/TinyStories", None, "train", "text"),
}


def iter_rows_api(path, name, split, field, page=100, workers=2):
    """Yield documents in order via datasets-server /rows (100 rows per request)."""
    base = "https://datasets-server.huggingface.co/rows?"

    def fetch(offset):
        q = dict(dataset=path, split=split, offset=offset, length=page)
        if name:
            q["config"] = name
        for attempt in range(12):  # the public API rate-limits (HTTP 429): back off
            try:
                with urllib.request.urlopen(base + urllib.parse.urlencode(q), timeout=120) as r:
                    return [row["row"][field] for row in json.load(r)["rows"]]
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(120, 5 * 2 ** attempt))
        return []

    offset = 0
    with ThreadPoolExecutor(workers) as ex:
        while True:
            pages = list(ex.map(fetch, range(offset, offset + page * workers, page)))
            for docs in pages:
                yield from docs
            offset += page * workers


PARQUET_PATTERNS = {
    "fineweb-edu": "sample/10BT/*.parquet",
    "tinystories": None,
}


def prepare_parquet(args, path, field, out_dir):
    """Download parquet shards, tokenize in parallel, write val then train."""
    import fnmatch
    from multiprocessing import Pool

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    pattern = PARQUET_PATTERNS[args.dataset]
    files = sorted(f for f in HfApi().list_repo_files(path, repo_type="dataset")
                   if fnmatch.fnmatch(f, pattern))
    need = args.val_tokens + args.train_tokens
    val_f = open(os.path.join(out_dir, "val.bin.tmp"), "wb")
    train_f = open(os.path.join(out_dir, "train.bin.tmp"), "wb")
    n_val = n_train = 0
    with Pool(args.workers) as pool:
        for fi, fname in enumerate(files):
            local = hf_hub_download(path, fname, repo_type="dataset", cache_dir=args.cache_dir)
            # split each shard by row group so all workers stay busy
            n_rg = pq.ParquetFile(local).metadata.num_row_groups
            chunks = [(local, rg) for rg in range(n_rg)]
            for toks in pool.imap(_encode_row_group, [(c, field) for c in chunks]):
                if n_val < args.val_tokens:
                    k = min(len(toks), args.val_tokens - n_val)
                    toks[:k].tofile(val_f)
                    n_val += k
                    toks = toks[k:]
                k = min(len(toks), args.train_tokens - n_train)
                toks[:k].tofile(train_f)
                n_train += k
                print(f"\rshard {fi + 1}/{len(files)} val {n_val / 1e6:.0f}M "
                      f"train {n_train / 1e9:.3f}B / {args.train_tokens / 1e9:.3f}B", end="", flush=True)
                if n_val + n_train >= need:
                    break
            if args.delete_shards:
                os.remove(os.path.realpath(local))
            if n_val + n_train >= need:
                break
    val_f.close()
    train_f.close()
    os.replace(os.path.join(out_dir, "val.bin.tmp"), os.path.join(out_dir, "val.bin"))
    os.replace(os.path.join(out_dir, "train.bin.tmp"), os.path.join(out_dir, "train.bin"))
    print(f"\nwrote {out_dir}: val {n_val / 1e6:.1f}M, train {n_train / 1e9:.3f}B tokens")


def _encode_row_group(args):
    (path, rg), field = args
    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    texts = pq.ParquetFile(path).read_row_group(rg, columns=[field]).column(field).to_pylist()
    out = [np.asarray(t + [enc.eot_token], dtype=np.uint16) for t in enc.encode_ordinary_batch(texts)]
    return np.concatenate(out) if out else np.empty(0, np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fineweb-edu", choices=DATASETS)
    ap.add_argument("--train-tokens", type=int, default=200_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--source", default="auto", choices=["auto", "datasets", "rows", "parquet"])
    ap.add_argument("--workers", type=int, default=32, help="tokenizer processes (parquet)")
    ap.add_argument("--cache-dir", default=None, help="HF download cache (parquet)")
    ap.add_argument("--delete-shards", action="store_true", help="delete parquet after use")
    args = ap.parse_args()

    import tiktoken

    path, name, split, field = DATASETS[args.dataset]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    source = args.source
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if source == "parquet":
        prepare_parquet(args, path, field, out_dir)
        return
    if source == "auto":
        try:
            import datasets  # noqa: F401
            source = "datasets"
        except ImportError:
            source = "rows"
    if source == "datasets":
        from datasets import load_dataset
        ds = (row[field] for row in load_dataset(path, name=name, split=split, streaming=True))
    else:
        ds = iter_rows_api(path, name, split, field)

    # Validation first, so the val split is a fixed, disjoint prefix of the stream.
    targets = [("val", args.val_tokens), ("train", args.train_tokens)]
    it = iter(ds)
    for split_name, budget in targets:
        arr = np.empty(budget, dtype=np.uint16)
        n = 0
        batch = []
        while n < budget:
            batch.append(next(it))
            if len(batch) < 256:
                continue
            for toks in enc.encode_ordinary_batch(batch):
                toks.append(eot)
                k = min(len(toks), budget - n)
                arr[n : n + k] = toks[:k]
                n += k
                if n >= budget:
                    break
            batch = []
            print(f"\r{split_name}: {n / 1e6:.1f}M / {budget / 1e6:.1f}M tokens", end="", flush=True)
        arr.tofile(os.path.join(out_dir, f"{split_name}.bin"))
        print()
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
