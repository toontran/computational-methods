# D0 synthesis

Date: 2026-04-28
Variant: D0 = S6 * relH1(A_cur v), multiplicative row-concentration guard.

## Verdict

**KILL.** D0 fails the reframed FAM-02 high-sample-entropy acceptance:
cos1^2 must be at least as good as S6 on every HIGH-sample-entropy matrix
and strictly better on at least one S6 failure. D0 does improve
diffuse-diffuse, but it regresses mixed-tail-balanced, mixed-tail-sharp,
and static-cex, with large slot-2 losses on mixed-tail-sharp and static-cex.

D1 and D2 remain spec-only and out of scope for this D0 closure.

## Evidence

T1 gradient check passed in
`summary/score_family_row_concentration_guard/variants/D0/diagnostics/T1_grad_check.txt`
with D0 relative errors <= 5.5e-10 on the recorded probes.

T2 per-block ranking is effectively unchanged versus S6 on the three
canonical probe matrices. The D0 oracle-minus-hm_triplet deltas differ from
S6 only at the relH1 product scale, and keep the same pass/fail pattern:
for example, mixed-tail-sharp remains negative at blocks 1, 12, and 31;
static-cex remains negative at blocks 1, 6, 12, and 31; diffuse-diffuse
remains negative at blocks 2, 6, and 31. D0 does not introduce a new T2
row-cheat improvement.

T3 completed outputs available under
`summary/score_family_row_concentration_guard/variants/D0/bench/`:

| matrix | S6 cos0 | S6 cos1 | D0 cos0 | D0 cos1 | delta cos1 | S6 tail | D0 tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffuse-diffuse | 0.865486 | 0.070462 | 0.746782 | 0.210003 | +0.139541 | 0.622985 | 0.699108 |
| mixed-tail-balanced | 0.828990 | 0.019488 | 0.829729 | 0.009973 | -0.009515 | 0.656197 | 0.655726 |
| mixed-tail-sharp | 0.856396 | 0.111796 | 0.873462 | 0.023998 | -0.087798 | 0.627044 | 0.618244 |
| static-cex | 0.967299 | 0.157249 | 0.984603 | 0.046809 | -0.110440 | 0.519802 | 0.514183 |

This is enough to kill D0 under the ship criterion: static-cex and
mixed-tail-sharp regress materially, so the high-entropy no-regression bar
is not met even though diffuse-diffuse improves.

## Missing-run attempts

The requested missing T3 reruns were started with the same policy set used
by the completed logs:

```
python half_window_sliding_hmean_experiment.py --matrix etf-basket-basis --half-win 32 --policies isvd combined future_hmean_r_sk_g --rsk-variant D0 --json-out summary/score_family_row_concentration_guard/variants/D0/bench/etf-basket-basis_win64.json --csv-out summary/score_family_row_concentration_guard/variants/D0/bench/etf-basket-basis_win64.csv --text-out summary/score_family_row_concentration_guard/variants/D0/bench/etf-basket-basis_win64.txt
python half_window_sliding_hmean_experiment.py --matrix mixed-tail-soft --half-win 32 --policies isvd combined future_hmean_r_sk_g --rsk-variant D0 --json-out summary/score_family_row_concentration_guard/variants/D0/bench/mixed-tail-soft_win64.json --csv-out summary/score_family_row_concentration_guard/variants/D0/bench/mixed-tail-soft_win64.csv --text-out summary/score_family_row_concentration_guard/variants/D0/bench/mixed-tail-soft_win64.txt
```

Both produced no JSON/CSV/TXT and no log output before being stopped as
long-running CPU-bound processes. Because completed high-entropy matrices
already violate the no-regression criterion, these missing outputs cannot
change the D0 verdict. The residual-spiky-shocks and risk-residual-panel
runs were not relaunched after the D0 ship criterion was already failed.

## Stretch-goal notes

Residual-spiky-shocks and risk-residual-panel are stretch-goal evidence
only under the reframed workflow acceptance. D0 is killed on the
high-sample-entropy criterion before spiky-residual outcome matters. This
does not decide whether another row-concentration variant could help the
low-entropy/spiky-residual regime; it only refutes the parameter-free
`S6 * relH1(A_cur v)` multiplier as a shippable high-entropy fix.

## Interpretation

The D0 result refutes the simple "multiply by current-window relH1 at the
end" version of Q3. It can move the optimizer enough to help one diffuse
case, but it also redirects slot 2 away from S6's better choices on
static-cex and mixed-tail-sharp. The evidence points back toward the
structural plateau/rank-r work rather than another scalar rank-1
multiplicative epicycle.
