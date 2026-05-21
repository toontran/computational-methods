# E0 Sum-Of-Trace Spec

Date: 2026-04-29

## Objective

For `Z in Stiefel(d, 2)` restricted to `B_union`:

`u_X(Z) = ||A_X Z||_F^2 / ||A_X||_F^2`.

E0 score:

`Score_E0(Z) = u_sk(Z) + u_cur(Z) + u_fut(Z)`.

If `A_sketch` is empty, omit `u_sk`. No HM, GM, min, per-column score, or
per-vector aggregator is applied.

Equivalent trace form:

`Score_E0(Z) = trace(Z^T C_E0 Z)`,

`C_E0 = I_sk A_sketch^T A_sketch / ||A_sketch||_F^2
      + A_cur^T A_cur / ||A_cur||_F^2
      + A_fut^T A_fut / ||A_fut||_F^2`.

The score is Grassmann-invariant: `Score_E0(ZR) = Score_E0(Z)` for any
orthogonal `R`.

Edge conventions:

- Omit any source `X` with an empty matrix or non-positive Frobenius
  denominator.
- All active source weights are fixed at `alpha_X = 1`; there is no learned or
  matrix-specific weighting in E0.
- The diagnostic domain is the projected right row space
  `B_union = rowspan([A_sketch; A_cur; A_fut])`.
- Ties between equal-score subspaces are read by the S-2 principal-angle rule,
  not by orientation inside the two-plane.

## Prediction Before Coding

Expected S-2 sign:

`delta = Score_E0(Z_winner) - Score_E0(Z_oracle) > 0`.

Reason: E0 is a top-r aggregate normalized covariance objective. Unless the
projected oracle frame is already the top trace subspace of `C_E0` inside
`B_union`, the optimized winner should outscore it. The useful question is
alignment, not the sign alone.

Expected verdict:

- `diffuse-diffuse`: uncertain but likely still at risk of `FAIL-subspace`,
  because E0 keeps the same magnitude-only evidence that FAM-01 B0 could not
  turn into stable streaming recovery.
- `static-cex`: uncertain but likely still at risk of `FAIL-subspace`.
- If E0 improves `pa_cos2[0]` above `0.9` on either target matrix, it becomes a
  serious S-2 follow-up candidate even with positive delta.

## Acceptance Gates

Use the canonical S-2 two-part rule with `tau_align = 0.5` for the slot-r
candidate gate and `pa_cos2[0] >= 0.9` for the slot-1 ship signal:

- `PASS-aligned`: `min(pa_cos2) >= tau_align` and `delta <= 0`.
- `PASS-with-overshoot`: `min(pa_cos2) >= tau_align` and `delta > 0`.
- `FAIL-slot-r-only`: `pa_cos2[0] >= 0.9` and `pa_cos2[1] < tau_align`.
- `FAIL-subspace`: `pa_cos2[0] < 0.9`.

E0 is killed for a matrix if the free-frame S-2 verdict is `FAIL-subspace`.
The target-pair acceptance question is whether `diffuse-diffuse` and
`static-cex` move out of `FAIL-subspace`, where FAM-01 B0 cannot ship. T3 is
blocked unless the target-pair S-2 partition gives at least a slot-1-aligned
verdict and the same-evidence blocker from the FAM-01 B0 T3 closure is
explicitly resolved.

## Diagnostic Plan

T1 gradient check:

- Required before any acceptance S-2 or T3 conclusion.
- Use the Stiefel finite-difference harness from INFRA-02 with the analytic
  E0 gradient `2 C_E0 Z`.
- Acceptance is relative error `< 1e-7` at float64.
- Current wrapper status: S-2 is wired; E0-specific `--gradient-check` is not
  wired yet, so T1 is a next engineering step.

S-2 smoke:

- Light smoke is allowed to verify wrapper plumbing and sign behavior.
- Smoke is not an acceptance run because optimizer budget is intentionally low.

S-2 target pair:

Run S-2 before any T3:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --matrices diffuse-diffuse static-cex --n-starts 100 --max-iter 200 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_target_pair`

Then, only if the target pair is not `FAIL-subspace`, run the seven-matrix S-2
suite:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --n-starts 100 --max-iter 200 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_full_suite`

Do not run long T3 benches until the S-2 partition is read and the current
same-evidence blocker is resolved.
