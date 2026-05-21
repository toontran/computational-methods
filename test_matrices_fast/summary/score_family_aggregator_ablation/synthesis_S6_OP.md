# AB-02 synthesis: F-norm vs operator-norm weighting

Date: 2026-04-28
Backlog item: `summary/overview/score_family_workflow.txt` §5 [AB-02]
Variant: `S6_OP`, where each normalized response uses
`u_X = raw_X / sigma_max(A_X)^2` instead of the S6 Frobenius denominator.

## Verdict

**KILL.** Operator-norm weighting does not satisfy the AB-02 acceptance bar.
It improves `mixed-tail-balanced` and `diffuse-diffuse`, but it regresses
the required heavy-tailed/static counterexample test `static-cex` and also
regresses `mixed-tail-sharp` and `etf-basket-basis` on slot-2.

FAM-01/FAM-03 should keep the existing Frobenius-normalized HM3 weighting
unless a new rank-r-specific normalization hypothesis is written.

## Acceptance

Workflow acceptance: compare S6_HM3 against `u_X = raw_X/sigma_max(A_X)^2`
on the §6 table. Decide on op-norm only if it improves heavy-tailed matrices
(`static-cex` explicitly) without regressing diffuse.

T1/T2 were already complete before this closure:

- T1 gradient check passed at float64, rel <= 3.2e-10:
  `summary/score_family_aggregator_ablation/S6_OP_T1_gradient_check.log`.
- T2 per-block outputs exist in
  `summary/score_family_aggregator_ablation/S6_OP_T2*`.

This closure completed the missing T3 outputs and aggregation.

## T3 Results

Sliding mode, `half_win=32`, block 31. S6_HM3 is the F-norm baseline from
`summary/bench_matrix_sweep_r_sk_g_S6/`. Online is the value-only online
baseline from INFRA-10, not the oracle-aware ceiling.

| matrix | S6 cos0 | S6 cos1 | S6 tail | S6_OP cos0 | S6_OP cos1 | S6_OP tail | delta cos1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static-cex | 0.9673 | 0.1572 | 0.5198 | 0.9768 | 0.0942 | 0.5185 | -0.0631 |
| mixed-tail-sharp | 0.8564 | 0.1118 | 0.6270 | 0.8654 | 0.0892 | 0.6216 | -0.0226 |
| mixed-tail-balanced | 0.8290 | 0.0195 | 0.6562 | 0.8469 | 0.1109 | 0.6353 | +0.0914 |
| mixed-tail-soft | 0.9204 | 0.1467 | 0.5657 | 0.8560 | 0.1639 | 0.6202 | +0.0172 |
| diffuse-diffuse | 0.8655 | 0.0705 | 0.6230 | 0.8824 | 0.1418 | 0.6006 | +0.0714 |
| etf-basket-basis | 1.0000 | 0.8073 | 0.1741 | 1.0000 | 0.6787 | 0.2697 | -0.1286 |
| residual-spiky-shocks | 0.5774 | 0.5159 | 0.7002 | 0.6989 | 0.5137 | 0.6238 | -0.0023 |

Acceptance screen:

- `static-cex` delta cos1^2 = -0.0631, so the explicit heavy-tailed/static
  requirement fails.
- `diffuse-diffuse` improves on both cos0^2 and cos1^2, so the diffuse
  no-regression condition is fine, but it cannot rescue the failed
  static-cex criterion.
- `mixed-tail-balanced` improves strongly (+0.0914), while
  `mixed-tail-sharp` regresses (-0.0226). The op-norm effect is not stable
  across the mixed-tail family.
- `etf-basket-basis` regresses materially (-0.1286), making op-norm
  weighting a poor default for the high-entropy suite.

## Commands Run

Completed missing T3 runs:

```
python half_window_sliding_hmean_experiment.py --matrix etf-basket-basis --half-win 32 --policies future_hmean_r_sk_g --rsk-variant S6_OP --json-out summary/score_family_aggregator_ablation/etf-basket-basis_S6_OP_win64.json --csv-out summary/score_family_aggregator_ablation/etf-basket-basis_S6_OP_win64.csv --text-out summary/score_family_aggregator_ablation/etf-basket-basis_S6_OP_win64.txt
python half_window_sliding_hmean_experiment.py --matrix residual-spiky-shocks --half-win 32 --policies future_hmean_r_sk_g --rsk-variant S6_OP --json-out summary/score_family_aggregator_ablation/residual-spiky-shocks_S6_OP_win64.json --csv-out summary/score_family_aggregator_ablation/residual-spiky-shocks_S6_OP_win64.csv --text-out summary/score_family_aggregator_ablation/residual-spiky-shocks_S6_OP_win64.txt
python summary/score_family_aggregator_ablation/aggregate_S6_OP.py
```

The other five S6_OP T3 outputs were produced by the AB-02 worker before
handoff using the same filename pattern.

## Propagation

Backlog propagation: AB-02 is closed as done / KILLED in
`score_family_workflow.txt`.

Overview propagation: no separate `score_design_overview.txt` update is
needed. AB-02 tested a silent normalization assumption under §2bis (b.i);
the result is local to the in-family ablation and does not resolve a Q-list
item or change the main theory/regime story.

Toolkit propagation: no diagnostic was added or closed.
