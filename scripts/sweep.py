"""Run every variant in a variants file under one base config.

    python scripts/sweep.py --config configs/tiny.yaml --variants configs/variants.yaml
    python scripts/sweep.py --config configs/gpt124m.yaml --variants configs/variants.yaml \
        --gpus 0,1,2,3,4,5,6,7            # one variant per GPU, in parallel
    python scripts/sweep.py ... --nproc 8  # each variant uses 8 GPUs via torchrun

Runs that already have a summary.json are skipped, so the sweep can be resumed.
Extra ``key=value`` arguments are forwarded to every run (e.g. ``seed=1``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import yaml


def run_name(v, seed):
    kw = "".join(f"_{k}{val}" for k, val in sorted(v.get("kwargs", {}).items()))
    return f"{v.get('name') or v['connection'] + kw}_s{seed}"


def cmd_for(v, args, extra):
    ov = [f"connection={v['connection']}", f"run_name={v['_run']}"]
    ov += [f"connection_kwargs.{k}={json.dumps(val)}" for k, val in v.get("kwargs", {}).items()]
    ov += [f"{k}={val}" for k, val in v.get("overrides", {}).items()]
    ov += extra
    if args.nproc > 1:
        head = ["torchrun", f"--nproc_per_node={args.nproc}", "-m", "hcfactory.train"]
    else:
        head = [sys.executable, "-m", "hcfactory.train"]
    return head + ["--config", args.config] + ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", default="configs/variants.yaml")
    ap.add_argument("--only", default=None, help="comma-separated run names to run")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids for parallel runs")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    with open(args.config) as f:
        base = yaml.safe_load(f) or {}
    out_dir = next((e.split("=", 1)[1] for e in extra if e.startswith("out_dir=")),
                   base.get("out_dir", "runs"))
    seed = next((e.split("=", 1)[1] for e in extra if e.startswith("seed=")), base.get("seed", 0))
    with open(args.variants) as f:
        variants = yaml.safe_load(f)["variants"]
    todo = []
    for v in variants:
        v["_run"] = run_name(v, seed)
        if args.only and v["_run"] not in args.only.split(","):
            continue
        if os.path.exists(os.path.join(out_dir, v["_run"], "summary.json")):
            print(f"skip {v['_run']} (done)")
            continue
        todo.append(v)

    if args.dry_run:
        for v in todo:
            print("[dry] " + " ".join(cmd_for(v, args, extra)))
        return
    os.makedirs(out_dir, exist_ok=True)
    gpus = args.gpus.split(",") if args.gpus else [None]
    running: dict = {}  # gpu -> (proc, name)
    while todo or running:
        for g, (p, name) in list(running.items()):
            if p.poll() is not None:
                print(f"[done] {name} rc={p.returncode}")
                del running[g]
        free = [g for g in gpus if g not in running]
        while todo and free:
            v, g = todo.pop(0), free.pop(0)
            cmd = cmd_for(v, args, extra)
            print("[run] " + " ".join(cmd), flush=True)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = g
            log = open(os.path.join(out_dir, f"{v['_run']}.stdout"), "w")
            running[g] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT),
                          v["_run"])
        time.sleep(2)


if __name__ == "__main__":
    main()
