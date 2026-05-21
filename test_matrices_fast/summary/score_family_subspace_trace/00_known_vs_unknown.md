# FAM-03 Subspace-Trace: Known vs Unknown

Date: 2026-04-29

Scope: FAM-03 only. This family tests trace-form rank-r objectives under the
same S-2 frame oracle-vs-winner gate used by FAM-01-DIAG. It does not change
FAM-01, FAM-07, or FAM-09 files.

## Settled Context

- The decision target is oracle identifiability, not oracle balance. For rank-r
  proposals, the controlling screen is S-2: compare `Score(Z_oracle)` to the
  optimized `Score(Z_winner)` and read the result with principal-angle
  alignment under the two-part rule in `score_design_overview.txt`
  section 1quinquies.
- The search domain is `B_union = rowspan([A_sketch; A_cur; A_fut])`, a
  right-row-space subspace of `R^d`.
- FAM-01-DIAG already supplies the needed Stiefel optimizer shape, polar
  retraction, tangent projection, and output convention. INFRA-02 gradient
  checking passed for the canonical HM frame score.
- FAM-01 B0, the HM3 rank-r frame lift on the current magnitude-only evidence,
  is not a shipped score. Its S-2 result was partial, and its later T3 run
  showed slot-1 collapse on the mixed-tail high-entropy matrices.
- The current sequencing says FAM-03 E0 should be specified and S-2-prepped,
  with priority on the two FAM-01 B0 `FAIL-subspace` matrices:
  `diffuse-diffuse` and `static-cex`.

## Open Points

- Whether a pure trace-sum frame objective can flip `diffuse-diffuse` and
  `static-cex` out of `FAIL-subspace` under S-2.
- Whether the trace-sum objective is a meaningful structural change or just
  the same magnitude-only evidence failure with the HM aggregator removed.
- Whether any E0 S-2 pass would be enough to justify T3 after the 2026-04-29
  FAM-01 B0 T3 closure. Current overview text says same-evidence rank-r
  variants inherit the structural concern; treat E0 T3 as blocked unless S-2
  gives a strong contradictory signal and the overview/backlog are updated.
- E1 and E2 are not specified beyond family placeholders. They should remain
  inactive until E0's S-2 result is read.

## Diagnostics That Resolve The Open Points

- T1: finite-difference check for the E0 trace-sum gradient on Stiefel frames.
  For E0 this is a direct trace form and should use the same tangent-projected
  central-difference harness as INFRA-02.
- S-2: run the frame oracle-vs-winner screen for E0 at `b31`, first on
  `diffuse-diffuse static-cex`, then on the seven-matrix section-6 suite if the
  target pair is not `FAIL-subspace`.
- T3: do not run yet. It is gated on S-2 partition and on resolving the
  same-evidence blocker noted above.

