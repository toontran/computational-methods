# E0 Sum-Of-Trace Synthesis

Date: 2026-04-29

## Status

E0 is specified and S-2-prepped. It is not accepted and no T3 bench should run
yet.

## Exact Score

For `Z in Stiefel(d, 2)` inside `B_union`:

`Score_E0(Z) = u_sk(Z) + u_cur(Z) + u_fut(Z)`,

where:

`u_X(Z) = ||A_X Z||_F^2 / ||A_X||_F^2`.

Empty or zero-denominator sources are omitted. Equivalently:

`Score_E0(Z) = trace(Z^T C_E0 Z)`,

`C_E0 = sum_X A_X^T A_X / ||A_X||_F^2`.

The Euclidean gradient is:

`grad Score_E0(Z) = 2 C_E0 Z`.

Use the existing Stiefel tangent projection and polar retraction from the
FAM-01-DIAG / INFRA-02 frame probe.

## Expected Gap Sign

Expected before the run: `delta = Score_E0(Z_winner) - Score_E0(Z_oracle) > 0`.

Reason: E0 optimizes the top-r aggregate normalized covariance subspace. Unless
the projected oracle frame is already the top trace subspace of `C_E0` inside
`B_union`, the optimized winner should outscore the oracle. A positive gap alone
is diagnostic context; the acceptance read is the principal-angle verdict.

## Acceptance Gates

Use the canonical S-2 two-part rule:

- Candidate / ship-aligned: `PASS-aligned` or `PASS-with-overshoot`.
- Conditional candidate: `FAIL-slot-r-only`, because slot 1 is still aligned.
- Kill for E0: `FAIL-subspace`, because `pa_cos2[0] < 0.9`.

The priority target is the FAM-01 B0 `FAIL-subspace` pair:
`diffuse-diffuse` and `static-cex`. E0 must move those out of `FAIL-subspace`
before any T3 is justified.

## Smoke Result

Light smoke command run on 2026-04-29:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --quick --anchors free --n-starts 3 --max-iter 20 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_smoke_fam03_e0_light`

Result on `diffuse-diffuse`:

| matrix | anchor | oracle | winner | delta | pa_cos2[0] | pa_cos2[1] | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| diffuse-diffuse | free | 0.4246 | 1.0000 | +0.5755 | 0.774 | 0.003 | FAIL-subspace |

Read: the wrapper works and the expected positive gap appears, but the primary
diffuse-diffuse target is still `FAIL-subspace` in the light smoke. This does
not close E0, because `static-cex` still needs the normal target-pair run and
the diffuse result is only a smoke-budget result.

## Gradient And Diagnostic Plan

1. Wire or reuse an E0-specific T1 check with the INFRA-02 Stiefel FD harness.
   Acceptance: relative error `< 1e-7`.
2. Run the normal-budget target-pair S-2 screen:
   `python summary/score_family_subspace_trace/probe_E0_frame_gap.py --matrices diffuse-diffuse static-cex --n-starts 100 --max-iter 200 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_target_pair`
3. If both target matrices move out of `FAIL-subspace`, run the seven-matrix
   S-2 suite. If either remains `FAIL-subspace`, do not run T3; redirect to
   evidence augmentation or close E0 as not solving the target pair.

## Next Action

Implement the E0 T1 gradient-check hook or run the target-pair S-2 only after
documenting why the trace-form gradient sanity is covered by INFRA-02. The
normal-budget target-pair S-2 is the decision point; T3 remains blocked.
