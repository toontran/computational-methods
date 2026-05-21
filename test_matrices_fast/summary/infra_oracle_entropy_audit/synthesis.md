# DIAG-01 Oracle Entropy and Regime-Label Audit

Date: 2026-04-28

Verdict: **ship**. The diagnostic is implemented and the regime labels are now measured. The revised labels are heuristic measurements, not theorem status for S6/HM3/relH1.

## Formula

For a window `W` and unit vector `v`, row-response energy is `e_i = (Wv)_i^2`, `p_i = e_i / sum_j e_j`, `H = -sum_i p_i log(p_i)`, `relH1 = H / log(m)`, and `effective_support = exp(H)` rows. The table below uses `eff_frac = effective_support / m` on `visible = [A_cur; A_fut]` for `V_exact[:,k]`.

## Revised Regime Table

| matrix | previous | measured | oracle slot1 median eff_frac | oracle slot2 median eff_frac | oracle slot2 min eff_frac | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| diffuse-diffuse | HIGH | HIGH | 1.000 | 1.000 | 1.000 |  |
| etf-basket-basis | HIGH | HIGH | 0.657 | 0.637 | 0.621 |  |
| mixed-tail-balanced | HIGH | HIGH | 1.000 | 1.000 | 1.000 |  |
| mixed-tail-sharp | HIGH | HIGH | 1.000 | 1.000 | 1.000 |  |
| mixed-tail-soft | HIGH | HIGH | 1.000 | 1.000 | 1.000 |  |
| residual-spiky-shocks | LOW | BOUNDARY | 0.177 | 0.474 | 0.403 | boundary/correction: previous label not supported by oracle slot entropy |
| risk-residual-panel | LOW | LOW | 0.245 | 0.171 | 0.076 |  |
| static-cex | HIGH | HIGH | 1.000 | 1.000 | 1.000 |  |

Classification rule used for this audit: HIGH if both oracle slots have median visible `eff_frac >= 0.50` and slot-2 never drops below `0.25` on the probed blocks; LOW if oracle slot-2 median visible `eff_frac < 0.25`; otherwise BOUNDARY. This is a measurement convention for regime labels, not a model theorem.

## Boundary Cases

- residual-spiky-shocks: previous LOW -> measured BOUNDARY; oracle slot-2 median visible `eff_frac=0.474` (min 0.403).

## Candidate Evidence

Median visible `eff_frac` by matrix, slot, and candidate across probed blocks:

| matrix | slot | oracle | S6 opt | mgain/iSVD | combined | rowcheat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diffuse-diffuse | 1 | 1.000 | 0.998 | 0.265 | 0.500 | 0.016 |
| diffuse-diffuse | 2 | 1.000 | 0.955 | 0.314 | 0.500 | 0.016 |
| etf-basket-basis | 1 | 0.657 | 0.660 | 0.659 | 0.661 | 0.631 |
| etf-basket-basis | 2 | 0.637 | 0.570 | 0.560 | 0.560 | 0.501 |
| mixed-tail-balanced | 1 | 1.000 | 0.992 | 0.022 | 0.502 | 0.016 |
| mixed-tail-balanced | 2 | 1.000 | 0.444 | 0.033 | 0.500 | 0.016 |
| mixed-tail-sharp | 1 | 1.000 | 0.979 | 0.018 | 0.502 | 0.016 |
| mixed-tail-sharp | 2 | 1.000 | 0.351 | 0.019 | 0.500 | 0.016 |
| mixed-tail-soft | 1 | 1.000 | 0.996 | 0.131 | 0.501 | 0.016 |
| mixed-tail-soft | 2 | 1.000 | 0.936 | 0.047 | 0.500 | 0.016 |
| residual-spiky-shocks | 1 | 0.177 | 0.284 | 0.244 | 0.503 | 0.016 |
| residual-spiky-shocks | 2 | 0.474 | 0.462 | 0.188 | 0.499 | 0.016 |
| risk-residual-panel | 1 | 0.245 | 0.055 | 0.025 | 0.292 | 0.016 |
| risk-residual-panel | 2 | 0.171 | 0.077 | 0.026 | 0.374 | 0.016 |
| static-cex | 1 | 1.000 | 0.972 | 0.016 | 0.503 | 0.016 |
| static-cex | 2 | 1.000 | 0.361 | 0.016 | 0.500 | 0.016 |

## Outputs

- `summary/infra_oracle_entropy_audit/audit.csv`: long-form per matrix/block/slot/candidate/window table.
- `summary/infra_oracle_entropy_audit/audit.json`: same records as JSON.
- `summary/infra_oracle_entropy_audit/regime_summary.csv`: revised regime table inputs.

## Assumptions

- Blocks are `1,2,6,12,31` where feasible with `half_win=32`, `rank=2`, `n=1024`, preset `fast`, seed `0`.
- Snapshot/optimizer settings are the diagnostic defaults in `oracle_entropy_audit.py`: `q0=4`, `qmax=16`, `num_restarts=1`, `maxit=40`, `post_expansion_maxit=30`, `union_maxit=40`, and `union_random_starts=4`.
- Candidate entropy is measured on row responses, not on vector coefficients.
- The diagnostic includes exact oracle vectors for classification; this is an audit, not an operational score.
- S6/HM3/relH1 remain heuristic until TH-01/TH-02/TH-03 establish stronger status.
