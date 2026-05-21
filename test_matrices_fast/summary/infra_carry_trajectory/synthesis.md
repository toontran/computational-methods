# INFRA-06 carry-trajectory probe — synthesis

Date: 2026-04-28
Backlog item: `summary/overview/score_family_workflow.txt §5 [INFRA-06]`
Probe script: `carry_trajectory_probe.py` (repo root: `test_matrices_fast/`)

## Setup

For each matrix in the §6 suite (plus `risk-residual-panel`) we ran the
streaming algorithm in sliding mode at `half_win=32` (full window 64, 31
blocks at `n=1024`), and recorded after each block:

- `s_top` = `state.s[0]`
- `sk_F2_low` = `Σ state.s_i²` (rank-r CARRY Frobenius²)
- `spectral_concentration` = `state.s[0]² / sk_F2_low`
- `carry_drift` = `‖V_t V_t^T − V_{t−1} V_{t−1}^T‖_F` (NaN on b1)

Update rule used: iSVD on `M_gain = [B_top; A_cur]` followed by
`make_state` (left-projected operator SVD), the same path the bench
takes when committing the carry per block. Per the workflow note, the
carry trajectory is mostly invariant to the slot-2 score, so the
deterministic iSVD update gives a clean reproducible reference.
Rank `r = 2` throughout (matches the bench convention).

CSV per matrix: `{matrix}_win64.csv`. Plot: `trajectories.png`.

## Per-matrix profile (final-block snapshot)

| matrix                | s_top₁→s_top₃₁ | sk_F2_low₁→₃₁ | spec_conc | drift behavior                          |
| --------------------- | -------------: | ------------: | --------: | --------------------------------------- |
| mixed-tail-sharp      | 0.958→0.978    | 1.79→1.90     | ≈0.50     | small ≈0.18, periodic spikes ≈1.4 (b2,b4,b20) |
| mixed-tail-balanced   | 0.941→0.973    | 1.74→1.87     | ≈0.50     | small ≈0.27, periodic spikes ≈1.3       |
| mixed-tail-soft       | 0.925→0.972    | 1.70→1.87     | ≈0.50     | medium ≈0.34, decays gently             |
| static-cex            | 0.927→0.990    | 1.72→1.94     | 0.50→0.51 | mostly tiny ≈0.005, three large jumps ≈1.4–2.0 |
| diffuse-diffuse       | 0.913→0.982    | 1.67→1.91     | ≈0.50     | medium ≈0.34, monotone decay 1.3→0.34   |
| etf-basket-basis      | 4.72→28.07     | 30.8→976      | 0.72→0.81 | decays quickly 0.25→0.015               |
| residual-spiky-shocks | 0.818→1.017    | 1.33→1.94     | 0.50→0.53 | ≈0.4, no decay                          |
| risk-residual-panel   | 1.13→1.24      | 2.53→3.03     | ≈0.51     | medium ≈0.21, occasional spikes 0.5–0.65|

## Three regimes

### R1. Carry matures fast — strong rank-1, low residual drift
**etf-basket-basis** is the canonical example. `s_top` grows ~6× over
31 blocks (4.7→28), `sk_F2_low` grows ~30× (30→976), and
`spectral_concentration` climbs from 0.72 to 0.81 (close to its
rank-2 ceiling of 1.0). Carry drift decays from 0.25 at b2 to ~0.015
by b20 — the carry is essentially locked.

**Multiplier shape recommendation**: `s_top²` (or its log) is the
right family-B multiplier here. The carry IS the answer — early
upweight of `u_sk` is safe.

### R2. Carry barely settles — flat s_top growth, persistent drift
**diffuse-diffuse**, **mixed-tail-soft**, **residual-spiky-shocks**
look similar at the spec-conc level (≈0.50, near the rank-2 floor of
1/r=0.5, meaning the two top svals are roughly co-equal). For these,
`s_top` is nearly stationary (0.9-ish across all blocks), `sk_F2_low`
crawls upward by <15%, and `carry_drift` either decays slowly
(diffuse, soft) or never decays (spiky-shocks: still 0.32 at b31).

**Multiplier shape recommendation**: `s_top²` would be a near-
constant multiplier (no informational content), and
`spectral_concentration` is pinned to 0.5 (also no information). The
ONLY signal the carry-state offers in this regime is `carry_drift` —
specifically its decay trend. A multiplier of the form
`exp(-λ · carry_drift)` would appropriately *down-weight* `u_sk`
when the carry is still moving, and let it back in once drift
settles. For residual-spiky-shocks where drift never decays, the
multiplier should stay sub-1 throughout — equivalent to
"down-weight u_sk forever in this regime", which matches the empirical
observation that iSVD/hybrid beat online here (online's carry-aware
score over-trusts an unstable carry).

### R3. Latching carry — long flat phases broken by sharp jumps
**static-cex** is the cleanest case: `carry_drift` is 1e-3..1e-2 most
of the time, but at b3, b4, b8, b11 it jumps to ≈1.4–2.0 (close to
the maximum theoretical value `√(2r)=2`, i.e., the carry rotates by
nearly π/2 in a single block). Between jumps `s_top` is flat to 5
sig figs. `mixed-tail-sharp/balanced` show a milder version of the
same pattern (occasional ≈1.4 jumps, otherwise small drift).

**Multiplier shape recommendation**: this is where instantaneous
`carry_drift` shines as a multiplier — when a jump happens,
down-weight `u_sk` for that block (the carry is mid-rotation), then
let it back in on subsequent blocks where drift collapses to 1e-3.
A short-window EMA of carry_drift, or `1 / (1 + drift)`, would be the
practical form. `s_top²` would NOT capture this — it stays nearly
constant across the jumps because `s_top` is intrinsic to the matrix
structure, not the rotation.

## Recommendation for FAM-04

The probe says no single multiplier shape is universally appropriate:

- **Use `s_top²` (or geometric-mean of carried svals at rank-r)** on
  matrices where rank-1 dominance grows visibly with block (R1):
  etf-basket-basis is the only clear case in the suite, but this
  regime is exactly where the carry IS the structural signal.
- **Use `1 / (1 + carry_drift)` (or an EMA)** on matrices in R2 and R3:
  R3 needs the instantaneous form (jump detection), R2 needs the
  smoothed form (chronic instability detection). These two share a
  formula since R3 is just R2 with sparser drift events.
- **Down-weight `u_sk` when drift never decays** (residual-spiky-shocks,
  risk-residual-panel): the carry is unreliable in this regime; the
  same drift-based multiplier handles it automatically (drift stays
  high → multiplier stays low).
- **`spectral_concentration`** is informative only in R1 (where it
  grows from 0.5 toward 1) and pinned at 0.5 in R2/R3 — useful as a
  *gate* selecting between the two multiplier families above
  (e.g., `if spec_conc > 0.6: use s_top²; else: use drift-based`).

A single defensible default for FAM-04's first variant: a hybrid
multiplier of the form

    m(state) = (1 - α) + α · σ(spec_conc - τ) · s_top² · 1/(1 + λ·drift)

with `τ ≈ 0.55` (the empirical R1 separator), `λ` calibrated so
drift=0.3 (R2 typical) gives multiplier ≈0.5, and `α` controlling
overall strength. The `1-α` baseline guarantees u_sk is never zeroed
out (avoids a P4-style oracle-exclusion regression).

## Open follow-ups (parked, not blocking FAM-04)

- The probe uses iSVD as the per-block update so the trajectory is
  reproducible. For the operational form FAM-04 will see, the carry
  comes from the policy in use (S6 or its rank-r lift). On most
  matrices the difference is small (left-projected SVD on a near-
  optimal V is close to iSVD on M_gain), but on static-cex the
  policy-driven trajectory may show larger latching effects since the
  combined-score v can disagree more sharply with iSVD's v. If FAM-04
  surfaces sensitivity here, re-run the probe with `policy=
  future_hmean_r_sk_g/S6` (the script's `run_carry_trajectory` accepts
  a policy arg — the iSVD branch is the only one currently wired).
- At rank-r > 2, `spectral_concentration` becomes more discriminating
  (the floor 1/r drops), so its role as a regime selector strengthens.
  Re-run when FAM-01 ships rank-r.
