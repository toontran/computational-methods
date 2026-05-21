# FAM-01-DIAG T1 gradient check

Run date: 2026-04-28

This checks the analytic gradient for the canonical frame score `Score(Z)=HM(u_sk(Z), u_cur(Z), u_fut(Z))`, with `u_X(Z)=||A_X Z||_F^2/||A_X||_F^2`, using the INFRA-02 Stiefel finite-difference harness.

- Status: **PASS**
- Cells run: **28**
- Score-gradient failures: **0**
- Trace-sanity failures: **0**
- Worst score rel_err: **5.84e-08**
- Worst trace sanity rel_err: **1.20e-08**
- Wall time: **75.5 s**

| matrix | block | r | score | max_rel | sanity_rel |
|---|---:|---:|---:|---:|---:|
| diffuse-diffuse | 1 | 2 | 1.7866e-03 | 3.87e-10 | 6.50e-10 |
| diffuse-diffuse | 2 | 2 | 1.8340e-03 | 4.90e-10 | 1.47e-10 |
| diffuse-diffuse | 12 | 2 | 2.4525e-03 | 8.48e-10 | 1.74e-09 |
| diffuse-diffuse | 31 | 2 | 2.1763e-03 | 1.96e-09 | 2.00e-10 |
| etf-basket-basis | 1 | 2 | 1.8117e-03 | 1.86e-10 | 2.26e-10 |
| etf-basket-basis | 2 | 2 | 1.6197e-03 | 1.29e-09 | 2.36e-10 |
| etf-basket-basis | 12 | 2 | 1.4100e-03 | 9.33e-10 | 2.67e-10 |
| etf-basket-basis | 31 | 2 | 6.0608e-04 | 5.09e-10 | 1.98e-09 |
| mixed-tail-balanced | 1 | 2 | 2.0737e-03 | 1.01e-09 | 6.98e-10 |
| mixed-tail-balanced | 2 | 2 | 1.9308e-03 | 3.61e-10 | 7.83e-10 |
| mixed-tail-balanced | 12 | 2 | 1.4775e-03 | 1.63e-09 | 6.37e-10 |
| mixed-tail-balanced | 31 | 2 | 1.2730e-03 | 5.13e-10 | 7.67e-10 |
| mixed-tail-sharp | 1 | 2 | 1.9765e-03 | 5.84e-08 | 1.02e-09 |
| mixed-tail-sharp | 2 | 2 | 1.6493e-03 | 5.51e-10 | 1.13e-09 |
| mixed-tail-sharp | 12 | 2 | 2.0646e-03 | 7.42e-10 | 6.95e-10 |
| mixed-tail-sharp | 31 | 2 | 2.1168e-03 | 1.95e-09 | 2.32e-10 |
| mixed-tail-soft | 1 | 2 | 2.2138e-03 | 1.34e-09 | 3.34e-09 |
| mixed-tail-soft | 2 | 2 | 6.4908e-04 | 6.98e-10 | 1.20e-08 |
| mixed-tail-soft | 12 | 2 | 1.8126e-03 | 9.79e-10 | 7.53e-10 |
| mixed-tail-soft | 31 | 2 | 5.2211e-04 | 7.75e-10 | 1.18e-09 |
| residual-spiky-shocks | 1 | 2 | 1.7280e-03 | 1.37e-09 | 1.58e-09 |
| residual-spiky-shocks | 2 | 2 | 1.5748e-03 | 2.96e-09 | 1.78e-09 |
| residual-spiky-shocks | 12 | 2 | 1.8893e-03 | 5.79e-10 | 1.30e-09 |
| residual-spiky-shocks | 31 | 2 | 1.2369e-03 | 2.21e-09 | 3.55e-10 |
| static-cex | 1 | 2 | 1.9140e-03 | 1.37e-09 | 1.28e-09 |
| static-cex | 2 | 2 | 1.5770e-03 | 4.41e-10 | 3.49e-10 |
| static-cex | 12 | 2 | 1.3992e-03 | 3.65e-10 | 6.30e-10 |
| static-cex | 31 | 2 | 1.5395e-03 | 3.79e-10 | 6.08e-10 |
