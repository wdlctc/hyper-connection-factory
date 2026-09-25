"""Download + tokenize a corpus into flat uint16 shards.

    python -m hcfactory.prepare_data --dataset fineweb-edu --train-tokens 200_000_000
    python -m hcfactory.prepare_data --dataset tinystories --train-tokens 50_000_000

Writes ``data/<dataset>/{train,val}.bin`` with GPT-2 BPE tokens (vocab 50257).

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fineweb-edu", choices=DATASETS)
    ap.add_argument("--train-tokens", type=int, default=200_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--source", default="auto", choices=["auto", "datasets", "rows"])
    args = ap.parse_args()

    import tiktoken

    path, name, split, field = DATASETS[args.dataset]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    source = args.source
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

    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
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
