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
| mixed-tail-sharp | nan | 13.8x | 1.21x | 5.26x | 22.2x | 0.013 |
| mixed-tail-balanced | nan | 9.63x | 8.06x | 1.25x | 13.8x | 0.000 |
| mixed-tail-soft | nan | 4.40x | 1.60x | 4.30x | 20.0x | 0.022 |
| static-cex | nan | 1.05x | 4.59x | 10.3x | 26.6x | 0.025 |
| diffuse-diffuse | nan | 15.7x | 35.2x | 11.7x | 1.29x | 0.005 |
| etf-basket-basis | nan | 1.03x | 1.02x | 1.02x | 1.06x | 0.652 |
| residual-spiky-shocks | nan | 12.4x | 7.08x | 2.71x | 2.25x | 0.266 |
| risk-residual-panel | nan | 2.36x | 7.03x | 3.47x | 6.36x | — |

## Slot-2 (oracle_v2_proj) ratio_max per block

Slot-2 is where S6 actually fails on T3. If slot-2 oracle is even more
imbalanced than slot-1, the M4 hypothesis predicts S6 fails harder on
slot-2 than slot-1 — which it empirically does.

| matrix | b1 | b2 | b6 | b12 | b31 | S6 cos1² |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 9.81x | 9.25x | 9.61x | 57.1x | 0.013 |
| mixed-tail-balanced | nan | 9.72x | 7.21x | 2.59x | 5.23x | 0.000 |
| mixed-tail-soft | nan | 15.3x | 2.26x | 1.84x | 8.99x | 0.022 |
| static-cex | nan | 1.05x | 1.96x | 5.04x | 22.1x | 0.025 |
| diffuse-diffuse | nan | 6.32x | 5.84x | 2.20x | 2.43x | 0.005 |
| etf-basket-basis | nan | 28.2x | 172x | 968x | 786x | 0.652 |
| residual-spiky-shocks | nan | 6.48x | 2.65x | 4.69x | 1.73x | 0.266 |
| risk-residual-panel | nan | 5.01x | 3.03x | 2.79x | 11.9x | — |

## Slot-1 ratio_skg1 (u_sk / u_g1) per block

ratio_skg1 > 1 means sketch is over-rewarded vs current half-window.
ratio_skg1 < 1 means sketch is under-rewarded. Either direction breaks
HM3's smallest-link reading of oracle. Note: at b1 sketch is empty so
ratio_skg1 is NaN.

| matrix | b1 | b2 | b6 | b12 | b31 |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 0.07x | 1.18x | 5.23x | 21.2x |
| mixed-tail-balanced | nan | 0.11x | 0.12x | 1.25x | 13.4x |
| mixed-tail-soft | nan | 0.23x | 1.58x | 4.29x | 19.8x |
| static-cex | nan | 1.05x | 4.59x | 10.3x | 26.2x |
| diffuse-diffuse | nan | 0.06x | 0.03x | 0.09x | 0.78x |
| etf-basket-basis | nan | 0.97x | 1.02x | 1.02x | 1.06x |
| residual-spiky-shocks | nan | 0.08x | 0.14x | 0.47x | 1.56x |
| risk-residual-panel | nan | 0.58x | 0.14x | 0.40x | 2.73x |

## Slot-2 component breakdown at b31 (the streaming-bench terminal block)

| matrix | u_sk | u_g1 | u_g2 | ratio_max | S6 cos1² |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | 0.0087 | 0.4976 | 0.4737 | 57.1x | 0.013 |
| mixed-tail-balanced | 0.0877 | 0.4589 | 0.4449 | 5.23x | 0.000 |
| mixed-tail-soft | 0.8326 | 0.0937 | 0.0926 | 8.99x | 0.022 |
| static-cex | 0.9314 | 0.0429 | 0.0422 | 22.1x | 0.025 |
| diffuse-diffuse | 0.5541 | 0.2285 | 0.2286 | 2.43x | 0.005 |
| etf-basket-basis | 0.0013 | 1.0303 | 0.8978 | 786x | 0.652 |
| residual-spiky-shocks | 0.2281 | 0.3893 | 0.3949 | 1.73x | 0.266 |
| risk-residual-panel | 0.7668 | 0.1353 | 0.0645 | 11.9x | — |

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

