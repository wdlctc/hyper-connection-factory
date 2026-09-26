"""Summarise a width ladder: final val loss vs model size, per connection.

    python scripts/scaling.py runs/scaling --out results/scaling

Expects ``<root>/w<d>/<variant>_s<seed>/summary.json``. Writes ``scaling.md``,
``scaling.json``, ``loss_vs_params.png`` and ``delta_vs_params.png``
(Δ = variant − prenorm at the same width; negative = better). If a width has
several seeds of a variant, the mean is plotted and the std is shown as error bars.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/scaling")
    ap.add_argument("--baseline", default="prenorm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data: dict = {}  # variant -> width -> list[summary]
    for f in glob.glob(os.path.join(args.root, "w*", "*", "summary.json")):
        width = int(re.search(r"/w(\d+)/", f).group(1))
        s = json.load(open(f))
        name = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))
        data.setdefault(name, {}).setdefault(width, []).append(s)
    if args.baseline not in data:
        raise SystemExit(f"no {args.baseline} runs under {args.root}")
    widths = sorted({w for v in data.values() for w in v})

    def stat(runs):
        xs = [r["final_val_loss"] for r in runs]
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

    base = {w: stat(r)[0] for w, r in data[args.baseline].items()}
    rows = []
    for name, per_w in sorted(data.items()):
        for w, runs in sorted(per_w.items()):
            m, sd = stat(runs)
            rows.append(dict(variant=name, width=w, params=runs[0]["params_non_embedding"],
                             val_loss=m, std=sd, seeds=len(runs),
                             delta=m - base[w] if w in base else None,
                             tok_per_s=st.mean(r["tok_per_s"] for r in runs)))
    json.dump(rows, open(os.path.join(args.out, "scaling.json"), "w"), indent=2)

    # markdown: Δ vs baseline, one column per width
    names = sorted(data, key=lambda n: (n != args.baseline, n))
    head = "| variant | " + " | ".join(f"d={w}" for w in widths) + " |"
    lines = [head, "|---" * (len(widths) + 1) + "|"]
    for n in names:
        cells = []
        for w in widths:
            r = next((r for r in rows if r["variant"] == n and r["width"] == w), None)
            if r is None:
                cells.append("—")
            elif n == args.baseline:
                cells.append(f"{r['val_loss']:.4f}")
            else:
                cells.append(f"{r['delta']:+.4f}" if r["delta"] is not None else f"{r['val_loss']:.4f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    bparams = {r["width"]: r["params"] for r in rows if r["variant"] == args.baseline}
    lines.append("| *non-emb params* | " + " | ".join(
        f"{bparams[w] / 1e6:.1f}M" if w in bparams else "—" for w in widths) + " |")
    table = "\n".join(lines)
    open(os.path.join(args.out, "scaling.md"), "w").write(
        f"Row `{args.baseline}` = its val loss; other rows = Δ vs {args.baseline} at the same width.\n\n"
        + table + "\n")
    print(table)

    cmap = plt.get_cmap("tab10")
    for kind in ("loss", "delta"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, n in enumerate(names):
            pts = sorted((r["params"], r["val_loss"] if kind == "loss" else r["delta"], r["std"])
                         for r in rows if r["variant"] == n
                         and (kind == "loss" or r["delta"] is not None))
            if not pts:
                continue
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.4, capsize=2,
                        ls="--" if n == args.baseline else "-", color=cmap(i % 10), label=n)
        ax.set_xscale("log")
        ax.set_xlabel("non-embedding parameters")
        ax.set_ylabel("final val loss" if kind == "loss" else f"Δ val loss vs {args.baseline}")
        if kind == "delta":
            ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("HC scaling: " + ("loss vs size" if kind == "loss" else "gain vs size"))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{kind}_vs_params.png"), dpi=150)
        plt.close(fig)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
