"""README figures for the GPU scaling study.

    python scripts/plot_scaling_gpu.py runs/scaling_gpu --out results/scaling_gpu

Writes
  gain_vs_size.png  Δ val loss vs pre-norm at S / M / L (one line per method)
  tradeoff_L.png    quality (Δ) vs throughput at L
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
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated palette, light surface). A method keeps its
# colour and marker in every figure; MUDD-ppn shares MUDD's colour (dashed).
METHODS = [
    # key(s) in runs, label, colour, marker, linestyle
    (("muddformer",), "MUDDFormer", "#2a78d6", "o", "-"),
    (("muddformer-ppn",), "MUDDFormer + PrePostDANorm", "#2a78d6", "o", "--"),
    (("mhc-n4",), "mHC (n=4)", "#eb6834", "s", "-"),
    (("hc-dynamic-n4",), "HC dynamic (n=4)", "#1baf7a", "^", "-"),
    (("mhar-h4",), "MHAR (4 heads)", "#eda100", "D", "-"),
    (("attnres-full",), "AttnRes", "#e87ba4", "v", "-"),
    (("hc-static-n4",), "HC static (n=4)", "#008300", "P", "-"),
    (("frac-dynamic-m2",), "Frac (m=2)", "#4a3aa7", "X", "-"),
]
SIZES = [("s", "S\n38.5M · 12 layers"), ("m", "M\n85M · 12 layers"), ("l", "L\n304M · 24 layers")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e6e5e0"


def load(root):
    out = {}  # (size, name) -> list[summary]
    for f in glob.glob(os.path.join(root, "*", "*", "summary.json")):
        size = os.path.basename(os.path.dirname(os.path.dirname(f)))
        name = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))
        out.setdefault((size, name), []).append(json.load(open(f)))
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def spread(ys, gap):
    """Nudge label y-positions apart (keeps order) so direct labels don't collide."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    shift = (sum(ys[i] for i in order) - sum(pos)) / len(pos)  # re-centre the block
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/scaling_gpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load(args.root)
    base = {s: st.mean(r["final_val_loss"] for r in data[(s, "prenorm")]) for s, _ in SIZES
            if (s, "prenorm") in data}
    base_tps = {s: st.mean(r["tok_per_s"] for r in data[(s, "prenorm")]) for s in base}
    xpos = {s: i for i, (s, _) in enumerate(SIZES)}

    # ---- gain vs size -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 6.0), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(0, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.08, 0.002, "pre-norm baseline", color=INK2, fontsize=8.5, va="bottom")
    # depth changes between M and L: mark it, since it confounds the M→L step
    ax.axvspan(1.5, 2.25, color="#efeee9", zorder=0)
    ax.text(1.875, 0.004, "depth 12 → 24 layers", color=INK2, fontsize=8.5, va="bottom", ha="center")

    ends, handles = [], []
    for keys, label, color, marker, ls in METHODS:
        pts = []
        for s, _ in SIZES:
            runs = sum((data.get((s, k), []) for k in keys), [])
            if not runs or s not in base:
                continue
            d = [r["final_val_loss"] - base[s] for r in runs]
            pts.append((xpos[s], st.mean(d), st.stdev(d) if len(d) > 1 else 0.0))
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.plot(xs, ys, ls=ls, color=color, lw=2, zorder=3)
        ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=8, color=color, mec=SURFACE, mew=1.5,
                    elinewidth=1.2, capsize=3, zorder=4)
        handles.append(matplotlib.lines.Line2D([], [], color=color, ls=ls, lw=2, marker=marker,
                                               ms=6, mec=color, label=label))
        if xs[-1] == 2:
            ends.append((ys[-1], label, color))
    # direct labels at the L end (values + names), spread to avoid collisions
    ly = spread([e[0] for e in ends], 0.0052)
    for (y, label, color), y2 in zip(ends, ly):
        ax.annotate(f"{y:+.3f}  {label}", xy=(2, y), xytext=(2.3, y2), textcoords="data",
                    fontsize=8.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=5))

    ax.set_xticks(range(len(SIZES)), [lab for _, lab in SIZES], color=INK)
    ax.set_xlim(-0.15, 3.35)
    ax.set_ylabel("Δ validation loss vs pre-norm  (lower is better)", color=INK, fontsize=10)
    ax.set_title("Gain over pre-norm at three model sizes (FineWeb-Edu, ~20 tokens/param)",
                 color=INK, fontsize=11.5, loc="left", pad=12)
    ax.set_ylim(-0.122, 0.012)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8,
              frameon=False, ncol=4, labelcolor=INK, handlelength=3.2)
    ax.spines["bottom"].set_bounds(0, 2)
    fig.text(0.01, 0.01, "Error bars: std over seeds where >1 seed (M: pre-norm, mHC, HC dynamic ×3; "
             "MUDD ×2). L: 1 seed each. Plain MUDDFormer diverged at L.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(args.out, "gain_vs_size.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- quality vs throughput at L ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.axhline(0, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.scatter([1.0], [0.0], s=70, color=INK2, marker="o", zorder=4)
    ax.annotate("pre-norm", (1.0, 0.0), xytext=(-8, 8), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK)
    for keys, label, color, marker, _ in METHODS:
        runs = sum((data.get(("l", k), []) for k in keys), [])
        if not runs:
            continue
        d = st.mean(r["final_val_loss"] for r in runs) - base["l"]
        sp = st.mean(r["tok_per_s"] for r in runs) / base_tps["l"]
        ax.scatter([sp], [d], s=80, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        ax.annotate(label, (sp, d), xytext=(7, -3), textcoords="offset points",
                    fontsize=8.5, color=INK)
    ax.set_xlim(0.15, 1.12)
    ax.set_xlabel("training throughput relative to pre-norm (8×H100, DDP)", color=INK, fontsize=10)
    ax.set_ylabel("Δ validation loss vs pre-norm", color=INK, fontsize=10)
    ax.set_title("Size L (304M, 6B tokens): quality vs speed", color=INK, fontsize=11.5,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, "MHAR's throughput is limited by the current implementation "
             "(slow fused kernel under torch.compile), not by the method.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(args.out, "tradeoff_L.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out}/gain_vs_size.png, tradeoff_L.png")


if __name__ == "__main__":
    main()
