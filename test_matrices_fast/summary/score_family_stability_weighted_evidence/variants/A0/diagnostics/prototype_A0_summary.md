# FAM-09 A0 prototype summary

Command defaults: matrices=['static-cex', 'diffuse-diffuse', 'mixed-tail-sharp'], blocks=[1, 6, 12, 31], lambda=1.0, frac=0.5, n_subsamples=12, n=512, half_win=32.
Elapsed seconds: 92.71

This is a panel reranking diagnostic, not an optimizer or T3 bench.

| matrix | block | slot | raw winner | stab winner | oracle raw rank | oracle stab rank |
|---|---:|---:|---|---|---:|---:|
| diffuse-diffuse | 1 | 1 | s6_opt | oracle | 2 | 1 |
| diffuse-diffuse | 1 | 2 | oracle | oracle | 1 | 1 |
| diffuse-diffuse | 6 | 1 | oracle | oracle | 1 | 1 |
| diffuse-diffuse | 6 | 2 | oracle | oracle | 1 | 1 |
| diffuse-diffuse | 12 | 1 | s6_opt | s6_opt | 2 | 2 |
| diffuse-diffuse | 12 | 2 | s6_opt | s6_opt | 2 | 2 |
| diffuse-diffuse | 31 | 1 | mgain | combined | 5 | 5 |
| diffuse-diffuse | 31 | 2 | oracle | oracle | 1 | 1 |
| mixed-tail-sharp | 1 | 1 | s6_opt | oracle | 2 | 1 |
| mixed-tail-sharp | 1 | 2 | s6_opt | oracle | 2 | 1 |
| mixed-tail-sharp | 6 | 1 | s6_opt | oracle | 2 | 1 |
| mixed-tail-sharp | 6 | 2 | oracle | oracle | 1 | 1 |
| mixed-tail-sharp | 12 | 1 | s6_opt | s6_opt | 2 | 2 |
| mixed-tail-sharp | 12 | 2 | oracle | oracle | 1 | 1 |
| mixed-tail-sharp | 31 | 1 | mgain | oracle | 5 | 1 |
| mixed-tail-sharp | 31 | 2 | oracle | oracle | 1 | 1 |
| static-cex | 1 | 1 | s6_opt | oracle | 2 | 1 |
| static-cex | 1 | 2 | s6_opt | oracle | 2 | 1 |
| static-cex | 6 | 1 | s6_opt | s6_opt | 2 | 2 |
| static-cex | 6 | 2 | s6_opt | s6_opt | 2 | 2 |
| static-cex | 12 | 1 | s6_opt | s6_opt | 2 | 2 |
| static-cex | 12 | 2 | s6_opt | s6_opt | 2 | 2 |
| static-cex | 31 | 1 | sketch | mgain | 5 | 2 |
| static-cex | 31 | 2 | oracle | sketch | 1 | 2 |

## Aggregate oracle-rank movement

- improved: 8
- unchanged: 15
- worsened: 1
- mean(raw_rank - stab_rank): 0.500

Positive mean means the stability penalty improved the oracle panel rank.

## By-matrix movement

| matrix | panels | improved | unchanged | worsened | mean(raw_rank - stab_rank) |
|---|---:|---:|---:|---:|---:|
| diffuse-diffuse | 8 | 1 | 7 | 0 | 0.125 |
| mixed-tail-sharp | 8 | 4 | 4 | 0 | 0.875 |
| static-cex | 8 | 3 | 4 | 1 | 0.500 |

Regression to track before optimizer work: `static-cex` block 31 slot 2 moves
from raw winner `oracle` to stability-weighted winner `sketch` (`oracle`
rank 1 -> 2). Verdict for A0: iterate, not ship.
