# Baseline Snapshot

FAM-03 compares against the FAM-01-DIAG canonical B0 S-2 partition, not against
new streaming numbers.

Canonical B0 S-2 result at b31:

| matrix | B0 S-2 verdict |
|---|---|
| residual-spiky-shocks | PASS-with-overshoot |
| etf-basket-basis | FAIL-slot-2-only |
| mixed-tail-sharp | FAIL-slot-2-only |
| mixed-tail-balanced | FAIL-slot-2-only |
| mixed-tail-soft | FAIL-slot-2-only with subspace-recheck flag |
| diffuse-diffuse | FAIL-subspace |
| static-cex | FAIL-subspace |

Primary E0 comparison target: flip `diffuse-diffuse` and `static-cex` out of
`FAIL-subspace` under the same S-2 two-part rule.

Source pointers:

- `summary/overview/score_design_overview.txt` section 1quinquies.
- `summary/overview/score_family_workflow.txt` FAM-01-DIAG and FAM-03 entries.
- `summary/infra_frame_oracle_gap/synthesis.md`.

