# E0 S-2 Smoke Notes

Date: 2026-04-29

This is a low-budget wrapper check, not an acceptance S-2 run:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --quick --n-starts 2 --max-iter 5 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_smoke`

Because the first wrapper version inherited upstream `--quick` behavior, this
ran the default seven-matrix S-2 list with `n_starts=2`, `max_iter=5`, and all
anchors. The wrapper has since been tightened so `--quick` without explicit
`--matrices` runs only `diffuse-diffuse`.

Immediate observations from the smoke output:

- The E0 wrapper works and emits the standard S-2 artifacts.
- The predicted positive gap is observed on every free-frame row in this
  low-budget smoke.
- `diffuse-diffuse` remains `FAIL-subspace` in the free-frame smoke
  (`pa2 ~= 0.774 / 0.003`), so the primary target is not solved by this quick
  check.
- `static-cex` looks promising in the free-frame smoke
  (`pa2 ~= 0.946 / 0.935`) despite positive gap, but this must be confirmed
  with the target-pair command at normal S-2 budget before any conclusion.

Verified quick wrapper command after the fix:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --quick --anchors free --n-starts 1 --max-iter 2 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_smoke_quick_verify`

Output: one `diffuse-diffuse` free-frame row, written under
`diagnostics/s2_smoke_quick_verify/`.

FAM-03 light smoke run on 2026-04-29:

`python summary/score_family_subspace_trace/probe_E0_frame_gap.py --quick --anchors free --n-starts 3 --max-iter 20 --out-dir summary/score_family_subspace_trace/variants/E0_sum_trace/diagnostics/s2_smoke_fam03_e0_light`

Output: one `diffuse-diffuse` free-frame row, written under
`diagnostics/s2_smoke_fam03_e0_light/`.

Result: oracle=0.4246, winner=1.0000, delta=+0.5755,
`pa_cos2=[0.774, 0.003]`. Under the canonical S-2 rule this is
`FAIL-subspace` because `pa_cos2[0] < 0.9`. This smoke confirms the wrapper and
expected positive gap sign, but it does not satisfy the target-pair acceptance
gate.
