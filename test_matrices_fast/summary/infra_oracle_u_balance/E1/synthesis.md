# DIAG-04 oracle u-balance audit — synthesis

Probe: `oracle_u_balance_audit.py` (per-block, half_win=32, rank=2).
Components: u_sk, u_g1, u_g2 from S6 (F-weighted HM3) at oracle_v_k_proj
for slot k ∈ {1, 2}. Ratios: max(u)/min(u), sk/g1, sk/g2, g1/g2.

Hypothesis (overview §1quater): HM3's "smallest-link" enforcer only
rewards oracle if oracle is roughly balanced under the chosen weights.
Frobenius weighting was inherited from §2(b)'s "unit-fixer" framing,
not calibrated against an oracle-balance criterion. This audit measures
the imbalance at scale and correlates it with S6 cos[k]² failure.

## Slot-1 (oracle_v1_proj) ratio_max per block

ratio_max = max(u_sk, u_g1, u_g2) / min(u_sk, u_g1, u_g2). 1.00x = perfect
balance; HM3 reads oracle as the high-score point. Large values mean
oracle's smallest u dominates HM3 and the score peak shifts off oracle.

| matrix | b1 | b2 | b6 | b12 | b31 | S6 cos1² |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 3.33x | 1.15x | 3.38x | 20.3x | 0.013 |
| mixed-tail-balanced | nan | 12.4x | 2.47x | 1.99x | 16.6x | 0.000 |
| mixed-tail-soft | nan | 5.06x | 1.28x | 2.18x | 16.9x | 0.022 |
| static-cex | nan | 1.21x | 5.06x | 11.3x | 26.3x | 0.025 |
| diffuse-diffuse | nan | 3.43x | 5.52x | 8.45x | 18.0x | 0.005 |
| etf-basket-basis | nan | 1.03x | 1.02x | 1.02x | 1.06x | 0.652 |
| residual-spiky-shocks | nan | 8.35x | 1.25x | 3.48x | 4.07x | 0.266 |
| risk-residual-panel | nan | 6.23x | 28.1x | 2.99x | 10.0x | — |

## Slot-2 (oracle_v2_proj) ratio_max per block

Slot-2 is where S6 actually fails on T3. If slot-2 oracle is even more
imbalanced than slot-1, the M4 hypothesis predicts S6 fails harder on
slot-2 than slot-1 — which it empirically does.

| matrix | b1 | b2 | b6 | b12 | b31 | S6 cos1² |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 4.34x | 5.27x | 2.05x | 9.24x | 0.013 |
| mixed-tail-balanced | nan | 6.18x | 9.60x | 20.5x | 9.20x | 0.000 |
| mixed-tail-soft | nan | 5.07x | 2.49x | 1.18x | 1.57x | 0.022 |
| static-cex | nan | 1.86x | 1.19x | 4.64x | 20.4x | 0.025 |
| diffuse-diffuse | nan | 2.40x | 2.20x | 1.69x | 7.00x | 0.005 |
| etf-basket-basis | nan | 9.54x | 207x | 1244x | 936x | 0.652 |
| residual-spiky-shocks | nan | 5.72x | 1.27x | 1.46x | 6.09x | 0.266 |
| risk-residual-panel | nan | 5.01x | 15.8x | 89.5x | 2.58x | — |

## Slot-1 ratio_skg1 (u_sk / u_g1) per block

ratio_skg1 > 1 means sketch is over-rewarded vs current half-window.
ratio_skg1 < 1 means sketch is under-rewarded. Either direction breaks
HM3's smallest-link reading of oracle. Note: at b1 sketch is empty so
ratio_skg1 is NaN.

| matrix | b1 | b2 | b6 | b12 | b31 |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 0.31x | 0.87x | 3.34x | 18.8x |
| mixed-tail-balanced | nan | 0.08x | 0.40x | 1.97x | 15.7x |
| mixed-tail-soft | nan | 0.20x | 0.78x | 2.17x | 16.5x |
| static-cex | nan | 0.84x | 5.06x | 10.2x | 26.3x |
| diffuse-diffuse | nan | 0.29x | 0.18x | 0.12x | 0.06x |
| etf-basket-basis | nan | 0.97x | 1.02x | 1.02x | 1.06x |
| residual-spiky-shocks | nan | 0.12x | 0.80x | 0.37x | 0.56x |
| risk-residual-panel | nan | 0.16x | 0.04x | 0.49x | 4.29x |

## Slot-2 component breakdown at b31 (the streaming-bench terminal block)

| matrix | u_sk | u_g1 | u_g2 | ratio_max | S6 cos1² |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | 0.0514 | 0.4748 | 0.4406 | 9.24x | 0.013 |
| mixed-tail-balanced | 0.0518 | 0.4769 | 0.4536 | 9.20x | 0.000 |
| mixed-tail-soft | 0.4296 | 0.2790 | 0.2730 | 1.57x | 0.022 |
| static-cex | 0.8971 | 0.0439 | 0.0441 | 20.4x | 0.025 |
| diffuse-diffuse | 0.7890 | 0.1128 | 0.1128 | 7.00x | 0.005 |
| etf-basket-basis | 0.0003 | 0.2964 | 0.2614 | 936x | 0.652 |
| residual-spiky-shocks | 0.7907 | 0.1298 | 0.1317 | 6.09x | 0.266 |
| risk-residual-panel | 0.3789 | 0.3399 | 0.1471 | 2.58x | — |

## Reading

Cross-matrix correlation between ratio_max and S6 cos1² failure.
The hypothesis predicts: matrices with large ratio_max at b31
(score peak shifted away from oracle) should have low S6 cos1².
Etf-basket-basis is the calibration anchor — if the hypothesis is
right, it should have the smallest ratio_max AND the highest S6 cos1².

Possible verdicts:
- VERIFIED if ratio_max ranks matrices in the same order as 1/cos1²
  AND ratio_max ≥ 5x on the failure matrices and < 2x on
  etf-basket-basis. AB-03 phase 2 is then strongly motivated.
- PARTIALLY VERIFIED if the rank-correlation is positive but a
  matrix has large ratio_max + decent cos1² (or vice versa). Mark
  it as a boundary case and consider whether other mechanisms
  (M2 carry pinning, §3 plateau drift) dominate there.
- REFUTED if ratio_max is uniformly small (< 2x everywhere) or
  rank-correlation with cos1² is near zero. AB-03 is killed at
  the audit stage.

Cross-references:
- score_design_overview.txt §1quater (M4 mechanism)
- score_design_overview.txt §2bis (b.iii) (calibration criterion)
- score_family_workflow.txt [DIAG-04] / [AB-03]
- diagnostic_toolkit.txt §6b (oracle u-imbalance signature)

