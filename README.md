# Hyper-Connection Factory

**One transformer body, every residual wiring.** Hyper-Connection Factory implements the
recent family of "beyond `x + f(x)`" residual connections for decoder LLMs behind a single
interface, trains them under an identical recipe, and reports the comparison.

Every variant shares **exactly** the same attention / MLP sublayers, initialisation, data,
optimizer and schedule. The only thing that changes is the `Connection` module: what each
sublayer reads and how its output is written back. So differences in the results are due
to the wiring and nothing else.

| key | method | paper | what changes | extra params | = pre-norm at init |
|---|---|---|---|---|---|
| `prenorm` | Pre-Norm residual | Xiong et al. 2020 | `h ← h + f(norm(h))` | 0 | — |
| `postnorm` | Post-Norm residual | Vaswani et al. 2017 | `h ← norm(h + f(h))` | 2L·d | ✗ |
| `hc` | Hyper-Connections (static / dynamic) | [Zhu et al., ICLR'25](https://arxiv.org/abs/2409.19606) | n parallel residual streams; learned read (A_m), write (B) and stream mixing (A_r) | n(n+2); dynamic adds d(n+2) + 2d (LayerNorm) + 2 | ✓ |
| `mhc` | Manifold-Constrained HC | [Xie et al. (DeepSeek), ICML'26](https://arxiv.org/abs/2512.24880) | HC with σ-gated read/write and a **doubly-stochastic** (Sinkhorn) stream-mixing matrix; learned read-out head | nd·n(n+2) + n(n+2) + 3 per sublayer, plus an n·nd read-out head | ✗ |
| `frac` | Frac-Connections | [Zhu et al., 2025](https://arxiv.org/abs/2503.14125) | HC without widening: split d into m fractions | 2m² + m; dynamic adds (d/m)(2m+3) + 2 | ✓ |
| `denseformer` | DenseFormer (DWA) | [Pagliardini et al., NeurIPS'24](https://arxiv.org/abs/2402.02622) | after each block, a learned static weighted average of all previous block outputs | O(L²) scalars | ✓ |
| `muddformer` | MUDDFormer | [Xiao et al., ICML'25](https://arxiv.org/abs/2502.12170) | **dynamic, per-token** dense weights, separately for Q, K, V and residual streams | small MLP per block | ✓ |
| `laurel` | LAuReL (RW / LR / RW+LR) | [Menghani et al., ICML'25](https://arxiv.org/abs/2411.07501) | learned residual weights and a low-rank skip path `x + xAB` | 2 + 2dr per sublayer | ✓ |
| `attnres` | Attention Residuals (Full / Block) | [Kimi Team, 2026](https://arxiv.org/abs/2603.15031) | **replace** the residual sum with softmax attention over previous sublayer outputs (depth attention) | 2d per sublayer (query + key-norm weight), plus one final router | ✗ (uniform avg) |
| `mhar` | Multi-Head Attention Residuals | [Luo et al., 2026](https://arxiv.org/abs/2607.27230) | AttnRes with H independent depth softmaxes (one per channel group) | 2d per sublayer, plus one final router | ✗ |
| `dar` | Delta Attention Residuals (per-sublayer / Block) | Luo et al., 2026 | keep the residual stream; **add** depth-routed deltas; optional zero-init gate | 2d per sublayer (+1 gate) | ✓ with `gate: zero` |

"= pre-norm at init" means that with the paper's initialisation the variant computes exactly the
same function as the pre-norm baseline. This is checked by
`tests/test_connections.py::test_equivalent_to_prenorm_at_init`.

## Results

### GPU scaling (8×H100): HC family and friends at S and M scale

FineWeb-Edu, GPT-2 tokenizer, ~20 training tokens per non-embedding parameter, global batch 262K
tokens, WSD schedule, bf16 + `torch.compile`, one variant per GPU. Configs: `configs/scaling_gpu/{s,m}.yaml`,
variants: `configs/variants_gpu.yaml`.

| size | shape | non-emb params | tokens |
|---|---|---|---|
| S | 12L × 512 | 38.5M | 0.79B |
| M | 12L × 768 | 85.0M | 1.70B |

Row `prenorm` shows its validation loss; the other rows show Δ vs pre-norm at the same size (negative = better).
Where there are several seeds the table gives mean ± std (pre-norm at M: 3 seeds, std 0.002).

| variant | s | m |
|---|---|---|
| prenorm | 3.4439 | 3.1993 ± 0.0021 |
| attnres-full | -0.0407 | -0.0130 |
| frac-dynamic-m2 | -0.0245 | -0.0150 |
| hc-dynamic-n4 | -0.0699 | -0.0457 ± 0.0042 (n=3) |
| hc-static-n4 | -0.0355 | -0.0075 |
| mhar-h4 | -0.0426 | -0.0292 |
| mhc-n4 | -0.0621 | -0.0495 ± 0.0029 (n=3) |
| muddformer | -0.1115 | -0.0698 ± 0.0011 (n=2) |
| *non-emb params* | 38.5M | 85.0M |

![gain vs size](results/scaling_gpu/delta_vs_params.png)

Full per-size tables with throughput and memory: [`results/scaling_gpu_s`](results/scaling_gpu_s),
[`results/scaling_gpu_m`](results/scaling_gpu_m).

What the GPU runs show:

* **Dynamic, input-dependent mixing is what survives scale.** At M, MUDDFormer (−0.070, 2 seeds),
  mHC (−0.050, 3 seeds) and dynamic HC (−0.046, 3 seeds) are 10–30× the pre-norm seed std.
  Static HC shrinks from −0.036 at S to −0.008 at M.
* **Gains shrink from S to M for every variant.** mHC shrinks least (−0.062 → −0.050). Whether
  the gains level off or vanish needs the L size (24L × 1024, 302M, `configs/scaling_gpu/l.yaml`),
  which has not been run yet.
* **Depth attention.** MHAR beats single-head AttnRes at M (−0.029 vs −0.013, 1 seed each).
* **Throughput** (relative to pre-norm, measured with 8 jobs sharing one node) ranges from 0.47×
  (HC/mHC) to 0.78× (Frac). MHAR's 0.23× is an **implementation** limit, not the method's:
  `torch.compile` fuses the multi-head RMSNorm backward and the softmax into one slow kernel.
  Contributions welcome.
* **Small scale misleads.** mHC was *worse* than pre-norm in the laptop runs below and static HC
  beat dynamic HC there. Both reverse on GPU.

### Laptop scale (sanity check only)

12 layers, d=256, 9.6M non-embedding params, 12.3M FineWeb-Edu tokens, Apple M3 Pro (fp32, MPS).
Pre-norm and MUDDFormer have 3 seeds, pre-norm's std is 0.006. Every other row is a single seed.
Throughput is measured on MPS.

| # | variant | val loss | Δ vs prenorm | params (non-emb) | +conn params | throughput (rel.) | 
|---|---|---|---|---|---|---|
| 1 | muddformer | 5.1240 ± 0.0333 | -0.1964 | 9.86M | 217.5K | 0.57× |
| 2 | muddformer-static | 5.2387 ± 0.0000 | -0.0818 | 9.65M | 0.3K | 0.69× |
| 3 | muddformer-r-only | 5.2447 ± 0.0000 | -0.0757 | 9.84M | 202.5K | 0.86× |
| 4 | mhar-h4 | 5.2574 ± 0.0000 | -0.0631 | 9.65M | 12.8K | 0.63× |
| 5 | hc-static-n4 | 5.2734 ± 0.0000 | -0.0470 | 9.64M | 0.6K | 0.81× |
| 6 | frac-dynamic-m2 | 5.2836 ± 0.0000 | -0.0369 | 9.66M | 21.8K | 0.94× |
| 7 | attnres-full | 5.2918 ± 0.0000 | -0.0286 | 9.65M | 12.8K | 0.57× |
| 8 | denseformer | 5.2929 ± 0.0000 | -0.0275 | 9.64M | 0.1K | 0.78× |
| 9 | dar-block | 5.2995 ± 0.0000 | -0.0209 | 9.65M | 12.3K | 0.82× |
| 10 | hc-dynamic-n4 | 5.3001 ± 0.0000 | -0.0203 | 9.69M | 49.8K | 0.69× |
| 11 | laurel-rw-lr | 5.3006 ± 0.0000 | -0.0198 | 9.84M | 196.7K | 0.82× |
| 12 | attnres-block | 5.3046 ± 0.0000 | -0.0158 | 9.65M | 12.8K | 0.87× |
| 13 | prenorm | 5.3204 ± 0.0055 | +0.0000 | 9.64M | 0.0K | 1.00× |
| 14 | dar | 5.3408 ± 0.0000 | +0.0204 | 9.65M | 12.3K | 0.66× |
| 15 | mhc-n4-idinit | 5.3507 ± 0.0000 | +0.0303 | 10.23M | 594.6K | 0.63× |
| 16 | mhc-n4 | 5.3551 ± 0.0000 | +0.0347 | 10.23M | 594.6K | 0.66× |
| 17 | postnorm-lr3e-4 | 5.9128 ± 0.0000 | +0.5924 | 9.64M | 6.1K | 0.89× |
| 18 | postnorm-lr1e-3 | 7.6286 ± 0.0000 | +2.3082 | 9.64M | 6.1K | 1.02× |
| 19 | postnorm | 7.6768 ± 0.0000 | +2.3564 | 9.64M | 6.1K | 1.08× |

* MUDDFormer's gain is robust across seeds (−0.196 ± 0.033). Its static-only and R-stream-only
  ablations each recover about 40% of it, so both ingredients matter.
* mHC and dynamic HC look weak here but are among the best on GPU (see above), so tiny-scale
  rankings do not transfer.
* Post-norm collapses to the unigram loss at lr ≥ 1e-3. At 3e-4 it trains but stays far behind
  pre-norm (5.91), and its LR has not been tuned further.

**Width ladder on MPS** (12L, d = 128…384, a fixed 12.3M tokens, LR ∝ 1/d; `configs/scaling/`).
Pre-norm at d=384 is *worse* than at d=256 because the larger models are undertrained on this
budget. That inflates every d=384 gain, so **do not read a scaling trend from this table**. The
GPU ladder above, where tokens grow with size, is the one to trust.

| variant | w128 | w256 | w384 |
|---|---|---|---|
| prenorm | 5.5014 | 5.3252 | 5.3427 |
| frac-dynamic-m2 | -0.0593 | -0.0416 | -0.1143 |
| hc-dynamic-n2 | — | -0.0602 | — |
| hc-dynamic-n4 | -0.1272 | -0.0250 | -0.1038 |
| hc-static-n2 | — | -0.0365 | — |
| hc-static-n4 | -0.0874 | -0.0518 | -0.0639 |
| hc-static-n8 | — | -0.0927 | — |
| mhc-n4 | -0.0160 | +0.0300 | -0.0377 |
| *non-emb params* | 2.6M | 9.6M | 21.2M |

## Quickstart

```bash
pip install -e .                      # pip >= 21.3; torch >= 2.4, numpy, pyyaml, tiktoken, matplotlib
pytest tests/                         # every variant: shapes, grads, causality, init-equivalence

# data: GPT-2 BPE tokens in flat uint16 shards
python -m hcfactory.prepare_data --dataset fineweb-edu --train-tokens 20_000_000 --val-tokens 1_000_000

# train one variant
python -m hcfactory.train --config configs/tiny.yaml connection=hc connection_kwargs.n=4

# train the whole comparison set, then build the table + plots
python scripts/sweep.py   --config configs/tiny.yaml --variants configs/variants.yaml
python scripts/compare.py runs/tiny --out results/tiny
```

### Using a connection in your own model

```python
from hcfactory import GPT, ModelConfig

cfg = ModelConfig(n_layer=12, d_model=768, n_head=12,
                  connection="mhar", connection_kwargs={"heads": 4})
model = GPT(cfg)
logits, loss = model(idx, targets)
```

## Training template

`hcfactory/train.py` is a single-file nanoGPT-style trainer shared by all variants:

* single device (CUDA / MPS / CPU) or multi-GPU DDP through `torchrun`
* bf16 autocast on CUDA, optional `torch.compile`, gradient accumulation
* AdamW. Connection parameters (gates, depth queries, mixing matrices) get **no weight decay**,
  because decaying them biases every variant back towards "no mixing". They can get their
  own LR multiplier (`connection_lr_mult`).
* cosine or WSD (warmup-stable-decay) schedule, gradient clipping, divergence detection
* evaluation on a **fixed** set of validation batches, identical for every variant
* logs: `runs/<sweep>/<run>/{config.json, log.jsonl, summary.json}`, plus optional W&B

Any config key can be overridden from the CLI (`key=value`, nested `a.b=value`).

| config | size | tokens | hardware |
|---|---|---|---|
| `configs/tiny.yaml` | 12L × 256d, 9.6M | 12.3M | laptop / 1 GPU, ~25 min per variant on MPS |
| `configs/gpt124m.yaml` | 12L × 768d, GPT-2 small | 2.6B | 8 GPUs (DDP) |
| `configs/gpt350m.yaml` | 24L × 1024d | 7.9B | 8 GPUs (DDP) |

```bash
# 8 GPUs, one variant per GPU in parallel (good for 124M)
python scripts/sweep.py --config configs/gpt124m.yaml --gpus 0,1,2,3,4,5,6,7 grad_accum=8
# or each variant on all 8 GPUs, one after another
python scripts/sweep.py --config configs/gpt124m.yaml --nproc 8
# multiple seeds: re-run with seed=1, seed=2; compare.py reports mean ± std
python scripts/sweep.py --config configs/gpt124m.yaml --nproc 8 seed=1
```

`scripts/bench.py` measures throughput and memory of every variant without any data.

## Adding a variant

```python
# hcfactory/connections/my_variant.py
from .base import Connection

class MyConnection(Connection):
    def __init__(self, cfg, alpha: float = 1.0):
        super().__init__(cfg)
        ...                                  # per-sublayer parameters

    def forward(self, x0, sublayers):        # x0: [B, T, D] token embedding
        h = x0
        for f in sublayers:                  # 2L sublayers: attn, mlp, attn, mlp, ...
            h = h + f(h)                     # f(x) = sublayer(norm(x))
        return h                             # goes to final norm + LM head
```

Register it in `hcfactory/connections/__init__.py`, add it to `tests/test_connections.py`
and `configs/variants.yaml`. Class attributes let a connection change the body:
`sublayer_prenorm = False` removes the sublayers' input norms (Post-Norm), and
`qkv_streams = True` lets attention take separate Q/K/V inputs (MUDDFormer).

## Implementation notes and fidelity

Each implementation follows the paper's equations and, where one exists, the official code.
The choices below are ones the papers leave open, so we decided them ourselves:

* **HC**: `layer_id` for the one-hot A_m init counts *sublayers* (attention and FFN each
  count as one layer, as in the paper). The dynamic branch uses LayerNorm (per the paper's
  pseudo-code). Streams are collapsed by summation.
* **mHC**: equations follow the paper and DeepSeek-V4's reference `inference/model.py`:
  norm-free RMS scaling of vec(H), pre `σ(·)`, post `2σ(·)`, row-softmax-first Sinkhorn
  (20 iters), and a learned sigmoid "HyperHead" read-out. The **initialisation of φ and b is
  not published**. We use φ ~ N(0, 0.02²) and b = 0 (uniform doubly-stochastic mixing at
  start). Both are exposed as kwargs (`phi_std`, `res_bias_diag`).
* **Frac-Connections**: m×2m mixing, LayerNorm over each d/m fraction, Y = A_r = I, B = 1 at init.
* **DenseFormer**: dilation supported, period fixed to 1.
* **MUDDFormer**: DA weights from `GELU(RMSNorm(X_i) W1) W2 + a_i` with W2 = 0 and a_i one-hot.
  The last layer produces only the R stream, with hidden width ×4. Not included:
  PrePostDANorm and the depth-varying FFN width.
* **LAuReL**: applied per sublayer. The RW weights are bounded as `(α, β) = 2·softmax(w)`
  (α = β = 1 at init). The paper asks for a bounding map but does not fix one. LR uses the
  paper's "column orthogonal" A init and B = 0. PA is not implemented.
* **AttnRes**: zero-init pseudo-query per sublayer, RMSNorm on keys, a final routing step
  over all sources before the output norm. Block mode keeps completed-block sums plus the
  current partial sum. The RMSNorm key weight is folded into the query (same function,
  normalised keys computed once per source).
* **DAR**: the Delta Block source is the true block delta `h_end − h_start`. The released code
  uses a recursive difference; see the DAR paper, appendix.

Found a discrepancy with a paper? Please open an issue. Faithful reproduction is the point.

## Repository layout

```
hcfactory/
  model.py                 shared Llama-style body (RoPE, SwiGLU, RMSNorm) + GPT wrapper
  connections/
    base.py                Connection interface, rms(), sinkhorn()
    residual.py            prenorm, postnorm
    hyper.py               hc, mhc, frac
    dense.py               denseformer, muddformer
    laurel.py              laurel
    depth_attention.py     attnres, mhar, dar
  train.py                 training template
  data.py, prepare_data.py data pipeline
configs/                   tiny / gpt124m / gpt350m + the variant list
scripts/                   sweep.py, compare.py, bench.py
tests/                     correctness tests for every variant
results/                   committed result tables and plots
```

## Citation

If you use this repository, please cite the original papers of the methods you compare
(links above), and optionally:

```bibtex
@software{hyperconnectionfactory2026,
  title  = {Hyper-Connection Factory: unified implementations and comparisons of residual-connection variants for LLMs},
  author = {Luo, Cheng},
  year   = {2026},
  url    = {https://github.com/wdlctc/hyper-connection-factory}
}
```

## License

MIT
