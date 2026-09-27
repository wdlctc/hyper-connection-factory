Row `prenorm` = its val loss; other rows = Δ vs prenorm at the same width.

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
