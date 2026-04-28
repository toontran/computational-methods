# INFRA-07 component time-series — synthesis

Probe: `component_time_series_probe.py` (per-block, half_win=32, rank=2).
Candidates: `oracle_v_proj`, `c_evi_v1`, `sketch_v1`, `combined_v1`, `mgain_svd_v1`.
Components: `(u_sk, u_g1, u_g2)` from S6 (F-weighted HM3) — see
score_design_overview.txt §2.

  u_sk(v) = ||A_sketch v||² / sk_F2_low      (rank-r CARRY F-norm²)
  u_g1(v) = ||A_cur   v||² / cur_F2
  u_g2(v) = ||A_fut   v||² / fut_F2

`u_sk` at block 1 is NaN by convention (sketch is empty; matches
`frob_hm3_score_diagnostic.py::f_hm_score`). `sketch_v1` is also undefined
at block 1 (no carry yet) so all three components are NaN there.

Cross-references: score_design_overview.txt §6 (matrix table outcomes),
diagnostic_toolkit.txt §6c (the cross-candidate component framing this
probe formalizes), §8(h) (the gap this closes).


## Per-matrix u_sk(sketch_v1) trajectory

`u_sk(sketch_v1)` measures how much rank-r CARRY mass the carry's top
singular vector itself captures. With rank=2, the ceiling is 1.0 (entire
mass on one direction); 0.5 is the equipartition floor (two singular
values of comparable magnitude).

| matrix                  |  b1  |   b2   |   b6   |  b12   |  b31   |
|-------------------------|------|--------|--------|--------|--------|
| mixed-tail-sharp        | nan  | 0.502  | 0.503  | 0.508  | 0.505  |
| mixed-tail-balanced     | nan  | 0.503  | 0.504  | 0.506  | 0.510  |
| mixed-tail-soft         | nan  | 0.501  | 0.501  | 0.505  | 0.515  |
| static-cex              | nan  | 0.501  | 0.505  | 0.508  | 0.508  |
| diffuse-diffuse         | nan  | 0.501  | 0.502  | 0.502  | 0.507  |
| etf-basket-basis        | nan  | 0.725  | 0.820  | 0.819  | 0.808  |
| residual-spiky-shocks   | nan  | 0.501  | 0.504  | 0.508  | 0.509  |
| risk-residual-panel     | nan  | 0.535  | 0.530  | 0.517  | 0.511  |

Reading: seven of eight matrices sit at the rank-2 equipartition floor
(~0.50) at every probed block — the two carried singular values are very
close. `etf-basket-basis` is the outlier (0.72 → 0.82): the basket
geometry produces a single dominant direction in the carry, so the top
carry SV captures most of the rank-r mass.


## Per-matrix u_sk(oracle_v_proj) trajectory

`u_sk(oracle_v_proj)` says how much of the carry's mass the population
top singular vector captures. A growing gap
`u_sk(sketch_v1) − u_sk(oracle_v_proj)` is the M2 signature
(score_design_overview.txt §1bis): the carry pins to *its own* top
direction even when that direction has drifted off the oracle.

| matrix                  |  b1  |   b2   |   b6   |  b12   |  b31   |
|-------------------------|------|--------|--------|--------|--------|
| mixed-tail-sharp        | nan  | 0.033  | 0.045  | 0.061  | 0.009  |
| mixed-tail-balanced     | nan  | 0.002  | 0.037  | 0.029  | 0.013  |
| mixed-tail-soft         | nan  | 0.019  | 0.165  | 0.224  | 0.450  |
| static-cex              | nan  | 0.153  | 0.357  | 0.432  | 0.480  |
| diffuse-diffuse         | nan  | 0.064  | 0.011  | 0.044  | 0.205  |
| etf-basket-basis        | nan  | 0.693  | 0.819  | 0.819  | 0.808  |
| residual-spiky-shocks   | nan  | 0.029  | 0.115  | 0.148  | 0.238  |
| risk-residual-panel     | nan  | 0.117  | 0.074  | 0.081  | 0.317  |


## Per-matrix u_sk(combined_v1) trajectory

For comparison: where the production combined-step optimizer's slot-1
sits in carry-space. By b31 this is essentially identical to
`sketch_v1` on every matrix — the combined picker has latched fully
onto the carry.

| matrix                  |  b1  |   b2   |   b6   |  b12   |  b31   |
|-------------------------|------|--------|--------|--------|--------|
| mixed-tail-sharp        | nan  | 0.251  | 0.415  | 0.467  | 0.489  |
| mixed-tail-balanced     | nan  | 0.251  | 0.428  | 0.467  | 0.490  |
| mixed-tail-soft         | nan  | 0.251  | 0.416  | 0.463  | 0.493  |
| static-cex              | nan  | 0.252  | 0.418  | 0.468  | 0.494  |
| diffuse-diffuse         | nan  | 0.251  | 0.417  | 0.460  | 0.483  |
| etf-basket-basis        | nan  | 0.713  | 0.819  | 0.819  | 0.808  |
| residual-spiky-shocks   | nan  | 0.254  | 0.419  | 0.468  | 0.486  |
| risk-residual-panel     | nan  | 0.251  | 0.393  | 0.425  | 0.497  |


## Qualitative reading

(1) **Rank-2 equipartition is the dominant carry shape.** Seven of the
eight matrices have `u_sk(sketch_v1) ≈ 0.5` at every block — the
streaming carry's two singular values are comparable. This is the
direct evidence behind §2bis (a.ii)'s observation that equal weighting
of `u_sk` ignores the rank-r structure of the carry: the rank-r CARRY
F-norm bakes in two near-equal directions, so `u_sk` is bounded above
by ~0.5 for any single-direction candidate.

(2) **u_sk(sketch_v1) − u_sk(oracle_v_proj) is the M2 signature.** On
`mixed-tail-sharp` and `mixed-tail-balanced` (the matrices where
combined fails slot-2 on the §6 table), oracle's `u_sk` stays at
0.01–0.06 — the oracle is essentially orthogonal to the rank-r carry
basis. The carry has pinned to a different rank-2 subspace, so the
score has nothing to attach the oracle direction to, exactly the M2
mechanism. Compare to `etf-basket-basis` where oracle and sketch_v1
have nearly identical `u_sk` (0.69 → 0.82) — the carry actually tracks
the population top SV here, and §6 reports S6 cos1² = 0.652 on this
matrix (the strong score-design signal).

(3) **Static-cex is intermediate.** `u_sk(oracle_v_proj)` climbs from
0.15 (b2) to 0.48 (b31) — the carry slowly aligns with the oracle.
But the carry is still rank-2 with two near-equal singular values
(`u_sk(sketch_v1) ≈ 0.50`), so `u_sk(oracle_v_proj)` plateaus around
0.48 (it captures one of the two directions, not both). This is the
score-design challenge in miniature: the oracle is reachable, but the
score has no way to distinguish the right direction within the rank-2
carry basis without lifting to Stiefel rank-r (Q1 in §7).

(4) **u_g1 and u_g2 carry the slot-1 selection signal.**
`cur_F2 ≈ fut_F2 ≈ N·rowscale²` is roughly block-invariant, so the
candidate ordering on `u_g1`/`u_g2` is the same shape every block:
oracle and `c_evi_v1` are near the per-block max on BOTH halves
(u_g1 ≈ u_g2 ≈ 0.014); `combined_v1` and `mgain_svd_v1` take the
A_cur top-SV (u_g1 ≈ 0.03, u_g2 ≈ 1e-6) — they have **no** A_fut
content. This is mechanism M3 from §1bis in concrete form: combined
is blind to A_fut so its slot-1 over-fits to the local A_cur draw.

(5) **diffuse-diffuse confirms M3.** `u_g2` for `combined_v1` is
~1e-6 at every block (the M3 signature), while oracle has u_g1 ≈
u_g2 ≈ 0.014 (HM2-balanced). Yet `u_sk(oracle_v_proj)` stays low
(0.06 → 0.20) — the carry still drifts off the oracle on this matrix,
which is why iSVD beats S6 on diffuse-diffuse (§6: iSVD 0.489 vs S6
0.005).

(6) **Spiky matrices show no extreme spike at the probed blocks.**
`residual-spiky-shocks` and `risk-residual-panel` have `u_g1`/`u_g2`
in the same range as the clean matrices at the probed blocks — the
spiky-row events likely happen at non-probed blocks. The candidate
ordering matches the clean cases. The differentiating effect of these
matrices (where iSVD and hybrid beat online) is not visible from this
component breakdown alone — it lives in the row-level distribution of
A_cur, which is what the deferred F1/relH1 guard is designed to catch.


## Cross-reference to §6 outcomes (score_design_overview.txt)

Matching the score-design line's empirical narrative:

- **Tail-dominant matrices where online dominates** (mixed-tail-sharp,
  mixed-tail-balanced): `u_sk(oracle_v_proj)` ≤ 0.07 at every probed
  block, while `u_sk(sketch_v1)` is at the rank-2 floor (~0.50). The
  S6 score has no signal to push the optimizer toward the oracle in
  carry-space — direct evidence for the P4 plateau pathology and Q1
  (rank-r lift) being the right structural fix.

- **Tail-dominant matrices where score-design has a foothold**
  (etf-basket-basis, mixed-tail-soft, static-cex): `u_sk(oracle_v_proj)`
  climbs steadily and lands at 0.45–0.81. On etf-basket-basis the
  oracle and sketch_v1 nearly coincide in carry-space — and indeed
  S6 reaches cos1² = 0.652 here (the cleanest score-design win).

- **Diffuse-diffuse**: M3 signature on `combined_v1` (`u_g2` ≈ 1e-6)
  while oracle has balanced u_g1/u_g2. iSVD's win on this matrix is
  outside what u_sk/u_g1/u_g2 reveal — it's an integrate-across-blocks
  effect.

- **Spiky-residual matrices**: components at the probed blocks look
  ordinary; the spiky behavior is row-level and aliased away by
  block-level Frobenius normalization.


## How to use this probe

Run before designing a new u_X form (toolkit §6c framing): pin the
candidate panel and read which matrix-block cells have the components
you expect to grow with carry maturity actually growing. The
`u_sk(oracle) - u_sk(sketch_v1)` gap per matrix is the most diagnostic
single number — it tells you whether the carry's u_sk ceiling is even
in a place the score can use.

Outputs:
- `{matrix}_components.csv` — block × candidate × (u_sk, u_g1, u_g2)
  plus the per-block Frobenius constants (sk_F2_low, cur_F2, fut_F2).
- `{matrix}_components.png` — three subplots (one per component);
  one line per candidate; symlog y-axis to show NaN → 1 dynamic range.
- `synthesis.md` (this file).
