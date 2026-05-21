# FAM-09: Stability-weighted evidence - family design

Date: 2026-04-29
Status: A0 diagnostic prototype started; optimizer integration not started.

## Hypothesis

Raw HM3 evidence rewards magnitude that is balanced across sketch/current/future
windows, but it does not ask whether that evidence is stable under row
perturbation inside a window. DIAG-03 shows that row-subsample CV, especially
`g2_p50_cv`, predicts carry-alignment decay beyond raw `u_g1` and `rel_h1`.

The FAM-09 hypothesis is that replacing each visible evidence value

```text
u_X(v) -> u_X(v) / (1 + lambda_X * CV_X(v))
```

shrinks fragile high-magnitude candidates and makes the HM optimum more
oracle-identifiable on the high-sample-entropy regime.

## Variant table

| variant | score | hyperparams | target | predicted effect | scope |
|---|---|---|---|---|---|
| **A0** | Diagnostic-only `HM(u_sk_stab, u_g1_stab, u_g2_stab)`, with `u_X_stab = u_X / (1 + lambda * CV_X)` | `lambda in {0, 0.5, 1, 2, 4}`, `p=0.50`, deterministic seed, small `n_subsamples` | Q15; FAM-09 feasibility | Re-ranks unstable non-oracle candidates downward on static-cex / diffuse-diffuse / mixed-tail-sharp blocks; no optimizer claim | THIS ROUND |
| A1 | Optimizer variant with cached deterministic row-partition CV for all three windows | `lambda`, partition count, all-window vs `g2`-only | high-entropy failures | Efficient derivative-free or piecewise-deterministic objective; ready for T1/T2 if A0 looks useful | LATER |
| A2 | Analytic variance proxy for row-energy evidence | `lambda`, window weights | same | Removes Monte-Carlo noise and enables analytic gradient if the proxy is smooth | LATER |

## A0 objective

For a unit candidate `v` at block `b`:

```text
u_X(v)       = ||A_X v||^2 / ||A_X||_F^2
CV_X(v)      = std_s(u_X(v; row_subsample_s)) / mean_s(u_X(v; row_subsample_s))
u_X^stab(v)  = u_X(v) / (1 + lambda * CV_X(v))
Score_A0(v)  = HM({u_X^stab(v)})
```

At block 1, sketch is absent, so A0 uses HM2 over current/future. At later
blocks it uses HM3 over sketch/current/future when sketch rows exist. A0 is
diagnostic-only: it ranks the existing DIAG-03 candidate panel and reports
alignment / rank changes.

## Acceptance proposal

- Prototype acceptance: produce a small-block CSV and summary for at least two
  high-entropy matrices, with raw HM and stability-weighted HM columns for the
  DIAG-03 candidate panel.
- Variant acceptance before T3: freeze the exact stability estimator; document
  derivative-free optimization or gradient theory; pass Tier-A S-1/S-2/S-3 on
  the §6 matrices at b31; then run T2 before any long T3 bench.
- Family acceptance: improve a high-entropy failure matrix without §6
  regression, with observed oracle-vs-winner gaps reported.

## Sequencing

1. Run A0 on a small block set using DIAG-03 helpers.
2. Inspect whether stability weighting moves oracle / S6 candidates upward
   relative to combined / mgain on high-entropy failures.
3. If yes, write A1 with a deterministic estimator and optimizer plan.
4. If no, kill or redirect to A2 only if the failure is estimator noise rather
   than absence of rank signal.
