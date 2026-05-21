# FAM-01-DIAG synthesis

Run date: 2026-04-28
Block: b31
Elapsed seconds: 356.490

Canonical screen: `hm` with `anchor=free`.
Optimizer: analytic retraction-based Stiefel gradient ascent.

Free-frame rows:
- diffuse-diffuse / fam03_e0_sum_trace: FAIL; delta=+0.575478; pa_cos2=[0.7742557502870547, 0.003386266325493544]
- residual-spiky-shocks / fam03_e0_sum_trace: FAIL; delta=+0.69688; pa_cos2=[0.4930348848892704, 2.956266328802142e-05]
- mixed-tail-soft / fam03_e0_sum_trace: FAIL; delta=+0.415537; pa_cos2=[0.9316753940509265, 0.13629598212581487]
- mixed-tail-sharp / fam03_e0_sum_trace: FAIL; delta=+0.494251; pa_cos2=[0.9298693553659709, 0.011253654650752398]
- static-cex / fam03_e0_sum_trace: FAIL; delta=+0.0557565; pa_cos2=[0.9462302626723317, 0.9346076586969696]
- mixed-tail-balanced / fam03_e0_sum_trace: FAIL; delta=+0.505103; pa_cos2=[0.903677341319257, 0.0018336794473493227]
- etf-basket-basis / fam03_e0_sum_trace: FAIL; delta=+0.168179; pa_cos2=[0.9962709233077454, 0.002583080564506917]
