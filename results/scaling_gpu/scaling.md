Row `prenorm` = its val loss; other rows = Δ vs prenorm at the same width.

| variant | s | m | l | xl |
|---|---|---|---|---|
| prenorm | 3.4439 | 3.1993 ± 0.0021 | 2.8405 | 2.4751 |
| attnres-full | -0.0407 | -0.0130 | -0.0318 | — |
| frac-dynamic-m2 | -0.0245 | -0.0150 | -0.0014 | — |
| hc-dynamic-n4 | -0.0699 | -0.0457 ± 0.0042 (n=3) | -0.0360 | — |
| hc-static-n4 | -0.0355 | -0.0075 | -0.0319 | — |
| mhar-h4 | -0.0426 | -0.0292 | -0.0378 | — |
| mhc-n4 | -0.0621 | -0.0495 ± 0.0029 (n=3) | -0.0396 | -0.0287 |
| muddformer | -0.1115 | -0.0698 ± 0.0011 (n=2) | — | — |
| muddformer-ppn | — | -0.0650 | -0.0710 | -0.0605 |
| *non-emb params* | 38.5M | 85.0M | 303.6M | 1214.4M |
