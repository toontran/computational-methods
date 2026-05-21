# E0 S-2 How To

The wrapper `summary/score_family_subspace_trace/probe_E0_frame_gap.py`
reuses `probe_frame_oracle_gap.py` and adds one runtime ablation:
`fam03_e0_sum_trace`.

It does not edit shared probe code.

## Smoke Run

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --quick --anchors free --n-starts 3 --max-iter 20 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_smoke`

`--quick` defaults to `diffuse-diffuse` in this wrapper when no explicit
`--matrices` list is supplied.

Smoke only checks wrapper plumbing and expected sign behavior. It is not an
acceptance run.

## Target Pair

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --matrices diffuse-diffuse static-cex --n-starts 100 --max-iter 200 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_target_pair`

## Full S-2 Suite

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --n-starts 100 --max-iter 200 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_full_suite`

## Reading The Output

Use the canonical two-part S-2 labels:

- `PASS-aligned`: `min(pa_cos2) >= 0.5` and `delta <= 0`.
- `PASS-with-overshoot`: `min(pa_cos2) >= 0.5` and `delta > 0`.
- `FAIL-slot-r-only`: `pa_cos2[0] >= 0.9` and `pa_cos2[1] < 0.5`.
- `FAIL-subspace`: `pa_cos2[0] < 0.9`.

The priority read is whether `diffuse-diffuse` and `static-cex` move out of
`FAIL-subspace`.

## Current Wrapper Limit

This wrapper is S-2 only. Do not use `--gradient-check` yet; the upstream
gradient-check path is wired to the canonical HM frame score, not to E0.

Before any T3 decision, either wire E0 into the INFRA-02 Stiefel finite-
difference check or explicitly certify the trace-form gradient
`grad = 2 C_E0 Z` against that harness.
