# AB-03 phase 1 synthesis: per-direction sigma² weighting (S6_E2)

Date: 2026-04-28
Backlog: `summary/overview/score_family_workflow.txt` §5 [AB-03] / [DIAG-04b]
Variant: `S6_E2` — per-direction weight `w_X(v) = sigma_X[k_X(v)]²` with
`k_X(v) = argmax_i (V_X[:,i]^T v)²` for each source X ∈ {sk, g1, g2}.
Implementation: `r_sk_g_score.py` `r_sk_g_value_grad(..., variant="S6_E2", e2_data=...)`.

## Verdict

**KILL.** Despite DIAG-04b verifying simultaneous oracle u-balance on
`diffuse-diffuse` (ratio_max(slot-2) = 2.43×) and `residual-spiky-shocks`
(1.73×), the operational T3 sliding bench at b31 shows S6_E2
**catastrophically regresses** cos1² on 4/7 §6 matrices (Δcos1² < −0.05),
including the two matrices DIAG-04b predicted would benefit. The
hypothesis "oracle-aware u-balance under per-direction reweighting
translates to bench cos1² gain" is **refuted at phase 1**.

The lesson: oracle u-balance is necessary but not sufficient. E2
balances the oracle but reshapes the score landscape so that the
optimizer no longer reaches the oracle (or reaches a different basin).

## T1 — gradient check

Pass at all probed blocks on `mixed-tail-sharp`. Max relative error
≤ 2.5e-10 across S6_E2 b1/b2/b12/b31, well below the 1e-7 threshold.
Justifies the locally-constant-argmax assumption: the FD stencil never
crossed a Voronoi-cell boundary at h=1e-6 on the probed v.

Log: `summary/score_family_aggregator_ablation/S6_E2_T1_gradient_check.log`.

## T3 — §6 sliding bench (half_win=32, block 31)

Reference baseline: `summary/bench_matrix_sweep_r_sk_g_S6/` (S6_HM3 = F-norm).

| matrix | S6 cos0 | S6 cos1 | S6 tail | E2 cos0 | E2 cos1 | E2 tail | Δcos0² | Δcos1² | Δtail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static-cex            | 0.9673 | 0.1572 | 0.5198 | 0.9731 | 0.0755 | 0.5237 | +0.0058 | **−0.0817** | +0.0039 |
| mixed-tail-sharp      | 0.8564 | 0.1118 | 0.6270 | 0.8608 | 0.0405 | 0.6287 | +0.0044 | **−0.0713** | +0.0016 |
| mixed-tail-balanced   | 0.8290 | 0.0195 | 0.6562 | 0.6655 | 0.0737 | 0.7758 | **−0.1634** | +0.0542 | +0.1196 |
| mixed-tail-soft       | 0.9204 | 0.1467 | 0.5657 | 0.9089 | 0.3229 | 0.5348 | −0.0115 | **+0.1762** | −0.0309 |
| diffuse-diffuse       | 0.8655 | 0.0705 | 0.6230 | 0.4715 | 0.0412 | 0.8880 | **−0.3940** | −0.0292 | +0.2650 |
| etf-basket-basis      | 1.0000 | 0.8073 | 0.1741 | 1.0000 | 0.7280 | 0.2350 | +0.0000 | **−0.0793** | +0.0609 |
| residual-spiky-shocks | 0.5774 | 0.5159 | 0.7002 | 0.6570 | 0.0903 | 0.7801 | +0.0797 | **−0.4257** | +0.0799 |

Catastrophic regressions (|Δcos1²| > 0.05): static-cex, mixed-tail-sharp,
etf-basket-basis, residual-spiky-shocks.

DIAG-04b correlation:
- `diffuse-diffuse` (E2 ratio_max slot-2 = 2.43×, predicted balance hit):
  Δcos1² = −0.029, Δcos0² = **−0.394**. The slot-1 axis collapsed.
- `residual-spiky-shocks` (E2 ratio_max slot-2 = 1.73×, also predicted hit):
  Δcos1² = **−0.426**. The largest regression in the suite, despite
  DIAG-04b's strongest balance signal.
- `mixed-tail-sharp` (E2 ratio_max slot-2 = 57×, predicted regression
  vs E1): Δcos1² = −0.071. Phase 1 confirms the predicted regression.

## Why DIAG-04b's prediction failed

Three working hypotheses (see overview §1quater for context):

1. **Balanced oracle is necessary but not sufficient.** HM3's
   "smallest-link enforcer" does require oracle balance under the
   weights used, but that only helps if the optimizer's basin lands
   ON the oracle. E2's per-direction weighting changes the score
   landscape away from the oracle by similar amounts: the new
   landscape's argmax is a different (non-oracle) point that the
   optimizer is happy to climb to instead.
2. **Per-direction weighting amplifies argmax noise.** When a candidate
   v projects similarly onto V_top[:,0] and V_top[:,1] (as on diffuse-
   spectra matrices), w_X(v) jumps discontinuously across the Voronoi
   boundary even though the oracle balance metric measured at the
   oracle alone is small. The optimizer chases these step changes.
3. **Slot-1 collapse on diffuse-diffuse / mixed-tail-balanced.** The
   per-direction weighting puts slot-1 (high-energy direction) and
   slot-2 (lower-energy direction) on equal footing in the score, so
   the deflation step that should pin slot-1 to V_state[:,0] no longer
   has a strong attraction — slot-1 cos0² drops by 0.16-0.39.

Implication: any per-direction reweighting that decouples slot-1
sigma² from the score risks slot-1 collapse on diffuse spectra. The
DIAG-04b oracle-balance metric does not see this because it evaluates
each slot independently AT the oracle.

## Recommendations

1. **KILL S6_E2** as a ship variant. Do not advance to phase 2 with E2
   alone. Per-direction weighting fails the §6 cos1² screen.
2. **Skip E3/E4** in this round. E3 (slot-aware scalar) has no advantage
   over E1 on the §6 high-entropy matrices that mattered, since
   DIAG-04b already showed E1 fails simultaneous balance everywhere.
   E4 (spectrally-weighted projection) introduces a free parameter
   without addressing the slot-1 collapse mechanism above.
3. **Promote FAM-01 (Stiefel rank-r lift) to active**. The per-vector
   reweighting failure on the §6 suite is consistent with the
   structural prediction in DIAG-04b: per-direction weighting alone
   cannot fix slot-2 imbalance on heavy-tailed/static matrices
   (static-cex stuck at >20× under all of current/E1/E2). The
   remaining lever is the rank-r lift that replaces sequential
   deflation greedy + rank-1 score with a true rank-r objective.
4. **Document the new principle**: "oracle u-balance under a weight
   scheme is necessary but not sufficient — the score *landscape*
   reachability also matters, and per-direction weighting can break
   reachability while improving balance." Should be added to
   `score_design_overview.txt` §1quater as a cautionary note.

## Cross-references

- Per-block CSVs: `summary/score_family_aggregator_ablation/S6_E2_*_win64.{json,csv,txt}`
- T1 log: `summary/score_family_aggregator_ablation/S6_E2_T1_gradient_check.log`
- T3 aggregate: `summary/score_family_aggregator_ablation/S6_E2_T3_aggregate.txt`
- DIAG-04b basis: `summary/infra_oracle_u_balance/scheme_comparison.md`
- Workflow item: `summary/overview/score_family_workflow.txt` [AB-03] / [DIAG-04b]
- Score landscape framing: `summary/overview/score_design_overview.txt` §1quater
