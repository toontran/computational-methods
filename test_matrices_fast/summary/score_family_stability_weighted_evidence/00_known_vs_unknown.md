# FAM-09: Stability-weighted evidence - known vs. unknown

Date: 2026-04-29
Pointers: `summary/overview/score_family_workflow.txt` [DIAG-03], [FAM-09];
          `summary/diag03_subsample_stability/synthesis.md`;
          `summary/overview/score_design_overview.txt` Q15;
          `summary/overview/diagnostic_toolkit.txt` §8(p).

This family starts the stability-weighted evidence track unblocked by DIAG-03.
It does not modify FAM-07 or FAM-03.

## Settled

- DIAG-03 shipped with SIGNAL. The canonical instability metric is
  `g2_p50_cv`, the CV of future-window evidence under 30 row-subsamples at
  `p=0.50`.
- DIAG-03 found `g2_p50_cv` predicts carry-alignment decay with `|r|=0.539`
  versus the best base predictor (`u_g1` or `rel_h1`) at `|r|=0.314`, a
  `+0.225` gap.
- The score-design redirect is now evidence augmentation, not another scalar
  aggregator over the same magnitude-only `u_X`. FAM-09 therefore changes the
  evidence inputs before the HM aggregation.
- The reusable sampler lives in
  `summary/diag03_subsample_stability/probe.py` as `_subsample_u` and
  `_summary_stats`.

## Open

- Whether a penalty calibrated from subsample CV improves a high-entropy
  failure matrix while preserving the §6 suite.
- Whether the score should penalize only `u_g2`, all visible windows
  (`u_sk`, `u_g1`, `u_g2`), or use asymmetric weights by window.
- Whether a Monte-Carlo CV estimate is stable enough for optimization. If not,
  FAM-09 needs an analytic or cached derivative-free objective before T3.
- Whether the stability penalty preserves oracle identifiability. Tier-A
  oracle-vs-winner checks remain the decision gate; a prettier diagnostic
  ranking alone is not acceptance.

## Toolkit gaps

- DIAG-03's sampler is diagnostic-grade, not optimizer-grade. It is stochastic
  and not currently differentiated.
- No cached block-level stability table is wired into the score pipeline.
  A practical A0 implementation should either freeze CV estimates per
  candidate/iteration seed or use a deterministic row partition estimate.

## Ship shape for this family

A shippable FAM-09 variant must define `u_X -> u_X * stability_X`, pass
gradient checks or explicitly use a derivative-free optimizer, pass Tier-A
oracle-identifiability screens, and improve at least one high-entropy failure
without regressing the §6 matrix suite.
