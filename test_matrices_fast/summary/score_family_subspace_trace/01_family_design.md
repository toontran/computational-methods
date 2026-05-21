# FAM-03 Family Design

Date: 2026-04-29

## Hypothesis

The HM-over-window-responses aggregator may be part of the rank-r
misidentification. FAM-03 replaces the HM frame aggregator with direct
subspace trace objectives:

`u_X(Z) = trace(Z^T A_X^T A_X Z) / ||A_X||_F^2 = ||A_X Z||_F^2 / ||A_X||_F^2`

for `Z in Stiefel(d, r)`, evaluated inside `B_union`.

This keeps the same visible magnitude evidence as FAM-01 B0 but changes the
frame objective from "balanced response across windows" to "captured normalized
trace mass." That makes E0 a narrow aggregator test, not an evidence
augmentation.

## Hypothesis Tags

- H1 argmax-shift: primary S-2 risk. E0 may simply move the argmax to the top
  aggregate trace subspace rather than to the oracle frame.
- H2 landscape discontinuity: not directly targeted; E0 is smooth.
- H3 slot-1-collapse: secondary target. Removing HM balance may restore a
  stronger aggregate-energy pull toward slot 1, but this is only meaningful if
  S-2 principal angles improve on `diffuse-diffuse` and `static-cex`.

## Variants

| variant | score | status | intended question |
|---|---|---|---|
| E0 sum-of-trace | `u_sk + u_cur + u_fut` over available windows | specified / S-2-prepped | Does a direct trace objective recover slot-1 alignment on the B0 `FAIL-subspace` pair? |
| E1 HM-of-traces | HM over trace components, with possible window groups | placeholder | Reserved; do not start until E0 S-2 is read. |
| E2 min-of-traces | smooth min over trace components | placeholder | Reserved; do not start until E0 S-2 is read. |

## Acceptance

Use the S-2 two-part rule:

- `PASS-aligned`: `min pa_cos2 >= tau_align` and `delta <= 0`.
- `PASS-with-overshoot`: `min pa_cos2 >= tau_align` and `delta > 0`.
- `FAIL-slot-r-only`: `pa_cos2[0] >= 0.9` and lower slot below the slot-r bar.
- `FAIL-subspace`: `pa_cos2[0] < 0.9`.

For E0, the key acceptance question is whether `diffuse-diffuse` and
`static-cex` stop being `FAIL-subspace`. T3 remains blocked until S-2 is known
and the same-evidence concern from FAM-01 B0 T3 is explicitly resolved.

