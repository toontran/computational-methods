# Stiefel FD gradient check (INFRA-02) — synthesis

Date: 2026-04-28
Backlog: `summary/overview/score_family_workflow.txt` §5 [INFRA-02]
Toolkit gap closed: `summary/overview/diagnostic_toolkit.txt` §8 (d)
Module: `test_matrices_fast/stiefel_grad_check.py`

## What this probe does

For the rank-r (Stiefel) lift of S6 / S6_GM, compares the analytic Euclidean gradient ∂S/∂V against a symmetric-difference quotient along column-seeded, polar-retracted tangent directions in T_V St. The test passes iff `rel < 1e-7` on every (matrix, block, r, variant) cell at float64.

A trace-form sanity check `f(V) = trace(V^T M V)` (analytic grad `2 M V`) is run alongside on every cell — if it fails (`rel ≥ 1e-7`), the FD-with-tangent-projection harness is broken, NOT the score implementation.

## Headline

- Cells run: **4** (matrices × blocks × ranks × variants)
- Score gradient cells passing (rel < 1e-7): **4/4**
- Trace-form sanity cells passing (rel < 1e-7): **4/4**
- Worst score rel_err: **2.29e-09**
- Worst sanity rel_err: **1.11e-09**
- Wall time: **44.0 s**
- Trace epsilon sweep: best **1.53e-09** at `eps=3e-5`; all swept eps values
  from `1e-4` through `1e-7` stayed below **7.23e-08**.

## Verdict: ship

All cells pass the acceptance bar at float64. INFRA-02 can be closed, and FAM-01 is no longer blocked on the Stiefel FD gradient-check infrastructure.

## Per-cell table

| matrix | block | r | variant | score | max_rel | sanity_rel | grad_tan_resid |
|---|---|---|---|---|---|---|---|
| mixed-tail-sharp | 2 | 2 | S6 | 1.2506e-03 | 1.27e-09 | 3.78e-10 | 4.22e-02 |
| mixed-tail-sharp | 2 | 2 | S6_GM | 1.5155e-03 | 2.29e-09 | 3.78e-10 | 7.74e-02 |
| mixed-tail-sharp | 12 | 2 | S6 | 2.7370e-03 | 7.47e-10 | 1.11e-09 | 1.87e-01 |
| mixed-tail-sharp | 12 | 2 | S6_GM | 2.7592e-03 | 7.91e-10 | 1.11e-09 | 1.64e-01 |

## Formulas

- Tangent space: `T_V St(n,r) = {Z : V^T Z + Z^T V = 0}`.
- Tangent projection: `P_V(G) = G - V sym(V^T G)`.
- Column-wise FD direction: draw `E_j` with non-zero entries only in column `j`, set `Z_j = P_V(E_j) / ||P_V(E_j)||_F`, then compare `(f(R_V(eps Z_j)) - f(R_V(-eps Z_j))) / (2 eps)` against `trace(P_V(G)^T Z_j)`.
- Trace sanity: `f(V) = trace(V^T M V)`, `G = 2 M V`, with `M` symmetric.
- Rank-r S6 lift: `u_X(V) = ||A_X V||_F^2 / ||A_X||_F^2`; aggregate with HM3 when sketch is present and HM2 at block 1.

## Files

- Module: `test_matrices_fast/stiefel_grad_check.py`
- Gauntlet table: `summary/infra_stiefel_fd_gradient_check/gauntlet.txt`
- Epsilon sweep: `summary/infra_stiefel_fd_gradient_check/epsilon_sweep.txt`
- This synthesis: `summary/infra_stiefel_fd_gradient_check/synthesis.md`

## Propagation

Diagnostic toolkit §2 / §8(d) and workflow INFRA-02 were updated in the closing patch. No `score_design_overview.txt` propagation is needed because this ships infrastructure only; it does not change a fundamental score-design Q or the heuristic status of S6/HM3/relH1.

## Notes for downstream consumers

- `stiefel_fd_check(score_value_grad_fn, V, ...)` is the public API. It accepts ANY value-and-grad function returning `(score, grad)` and runs the tangent-projection FD compare; reuse it for FAM-01 / FAM-03 as new variants land.
- `stiefel_tangent_project(V, G) = G - V sym(V^T G)` is exposed as a library function; `qr_retract(V, W)` is the QR retraction with a diag-sign fix to avoid spurious sign flips at small step sizes.
- For O(r)-invariant scores (S6 / S6_GM are subspace functions), the Euclidean gradient has no vertical `V * skew` component, but it can and usually does have a normal component removed by `P_V`. The reported `grad_tan_resid` is therefore diagnostic only, not an acceptance criterion.
