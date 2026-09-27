Row `prenorm` = its val loss; other rows = Δ vs prenorm at the same width.

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
