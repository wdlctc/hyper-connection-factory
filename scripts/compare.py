"""Aggregate a sweep into a results table + plots.

    python scripts/compare.py runs/tiny --out results/tiny

Writes ``results.md`` (markdown table), ``results.json`` and
``val_loss.png`` / ``train_loss.png`` / ``tradeoff.png``. When several seeds of
the same variant exist (run names ``<name>_s<seed>``), mean ± std is reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as f:
        s = json.load(f)
    logs = [json.loads(line) for line in open(os.path.join(run_dir, "log.jsonl"))]
    return s, logs


def ema(xs, a=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="prenorm")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join("results", os.path.basename(args.runs.rstrip("/")))
    os.makedirs(out, exist_ok=True)

    groups: dict[str, list] = {}
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        name = re.sub(r"_s\d+$", "", os.path.basename(d))
        groups.setdefault(name, []).append(load(d))
    if not groups:
        raise SystemExit(f"no finished runs in {args.runs}")

    rows = []
    for name, runs in groups.items():
        finals = [s["final_val_loss"] for s, _ in runs]
        s0 = runs[0][0]
        rows.append(dict(
            name=name, seeds=len(runs),
            val_loss=st.mean(finals), val_std=st.stdev(finals) if len(finals) > 1 else 0.0,
            params=s0["params_non_embedding"], conn_params=s0["connection_params"],
            tok_per_s=st.mean(s["tok_per_s"] for s, _ in runs),
            mem=s0.get("peak_mem_gb"), diverged=any(s["diverged"] for s, _ in runs),
            tokens=s0["tokens"],
        ))
    base = next((r for r in rows if r["name"] == args.baseline), None)
    rows.sort(key=lambda r: r["val_loss"])
    for r in rows:
        r["delta"] = r["val_loss"] - base["val_loss"] if base else float("nan")
        r["rel_speed"] = r["tok_per_s"] / base["tok_per_s"] if base else float("nan")

    multi = any(r["seeds"] > 1 for r in rows)
    lines = [
        "| # | variant | val loss | Δ vs " + args.baseline + " | params (non-emb) | +conn params "
        "| throughput (rel.) | " + ("peak mem | " if rows[0]["mem"] else "") + "",
        "|---|---|---|---|---|---|---|" + ("---|" if rows[0]["mem"] else ""),
    ]
    for i, r in enumerate(rows, 1):
        vl = f"{r['val_loss']:.4f}" + (f" ± {r['val_std']:.4f}" if multi else "")
        if r["diverged"]:
            vl += " (diverged)"
        line = (f"| {i} | {r['name']} | {vl} | {r['delta']:+.4f} | {r['params'] / 1e6:.2f}M | "
                f"{r['conn_params'] / 1e3:.1f}K | {r['rel_speed']:.2f}× |")
        if r["mem"]:
            line += f" {r['mem']:.1f} GB |"
        lines.append(line)
    table = "\n".join(lines)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(table)

    # ---- plots -----------------------------------------------------------
    order = [r["name"] for r in rows]
    cmap = plt.get_cmap("tab20")
    color = {n: cmap(i % 20) for i, n in enumerate(sorted(groups))}
    title = args.title or os.path.basename(args.runs.rstrip("/"))

    for key, fname, smooth in [("val_loss", "val_loss.png", False), ("loss", "train_loss.png", True)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        lo = []
        for name in order:
            _, logs = groups[name][0]
            pts = [(r["tokens"], r[key]) for r in logs if key in r]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ys = ema(ys, 0.95) if smooth else ys
            ls = "--" if name == args.baseline else "-"
            ax.plot([x / 1e6 for x in xs], ys, ls, label=name, color=color[name], lw=1.4)
            lo.append(ys[-1])
        ax.set_xlabel("tokens (M)")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(f"{title}: {key.replace('_', ' ')}")
        if lo:  # zoom into the tail where the variants separate
            ax.set_ylim(min(lo) - 0.05, min(lo) + 0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=150)
        plt.close(fig)

    if base:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            ax.scatter(r["rel_speed"], r["delta"], color=color[r["name"]], s=40)
            ax.annotate(r["name"], (r["rel_speed"], r["delta"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(1, color="gray", lw=0.8)
        ax.set_xlabel(f"throughput relative to {args.baseline}")
        ax.set_ylabel(f"Δ val loss vs {args.baseline} (lower is better)")
        ax.set_title(f"{title}: quality vs speed")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "tradeoff.png"), dpi=150)
        plt.close(fig)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
