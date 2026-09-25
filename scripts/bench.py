"""Throughput / memory micro-benchmark of every variant (random tokens, no data needed).

    python scripts/bench.py --config configs/tiny.yaml --steps 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hcfactory.model import GPT, ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--variants", default="configs/variants.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    variants = yaml.safe_load(open(args.variants))["variants"]
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    B, T = cfg.get("batch_size", 16), cfg.get("seq_len", 1024)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    print(f"device={dev} B={B} T={T}")
    print(f"{'variant':18s} {'params':>8s} {'+conn':>8s} {'tok/s':>9s} {'mem GB':>7s}")
    for v in variants:
        m = GPT(ModelConfig(n_layer=cfg["n_layer"], n_head=cfg["n_head"], d_model=cfg["d_model"],
                            max_seq_len=T, connection=v["connection"],
                            connection_kwargs=v.get("kwargs", {}))).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        f = torch.compile(m) if args.compile else m
        x = torch.randint(0, 50257, (B, T), device=dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i in range(args.steps + 3):
            if i == 3:
                sync()
                t0 = time.perf_counter()
            with amp:
                _, loss = f(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        sync()
        tps = B * T * args.steps / (time.perf_counter() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
        print(f"{v['name']:18s} {m.num_params() / 1e6:7.2f}M {m.connection_params() / 1e3:7.1f}K "
              f"{tps:9.0f} {mem:7.2f}")
        del m, opt, f


if __name__ == "__main__":
    main()
