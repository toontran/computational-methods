# FAM-07 T2 per-block synthesis

Command:

```
python summary/score_family_cross_window_consistency/03_t2_per_block_probe.py
```

Scope: matrices `mixed-tail-sharp`, `static-cex`, `diffuse-diffuse`, `etf-basket-basis`; blocks `1, 2, 12, 31`; half_win=32; rank=2; seed=0.

Artifacts:

- `04_t2_per_block.csv`
- `04_t2_per_block.json`

Rows: 144 total (96 vector, 48 frame).
T2 signature rows: 36 total (24 vector, 12 frame).
T2 signature cells: 12 of 16 matrix/block cells.

Acceptance read:

PASS for T2/K2: cross-window consistency adds diagnostic signal beyond HM3 on at least one real high-entropy cell.

Diagnostic value read:

The signal is strongest exactly where FAM-07 was aimed. `diffuse-diffuse`
and `static-cex` both fire on all four probed blocks, with high current/
future magnitudes but near-zero frame consistency on `frame_S6_greedy`.
`mixed-tail-sharp` also fires on all four blocks, mostly through the same
S6-greedy frame and v2 rows. `etf-basket-basis` has no signature rows,
which is a useful no-regress sanity check rather than a failure.

This clears only the T2/K2 diagnostic bar. The remaining gate is Tier-A
S-2 for the actual FAM-07 augmented frame score: before T3, run the
frame-level oracle-vs-winner screen and require the two-part S-2 verdict
to move `diffuse-diffuse` and/or `static-cex` out of FAIL-subspace, or
at minimum not introduce a new FAIL-subspace result on the mixed-tail
candidate set. T3 should not be interpreted until that S-2 screen exists.

Signature counts by matrix:

- `diffuse-diffuse`: 12
- `mixed-tail-sharp`: 12
- `static-cex`: 12

Strongest signature rows (lowest rho first):

- `mixed-tail-sharp` b1 `frame_S6_greedy_v2` (vector): rho=-0.0124, u_g1=1.7516e-02, u_g2=1.5098e-02, score_HM3=1.6217e-02, score_FAM07=2.4840e-06
- `static-cex` b31 `frame_S6_greedy_v2` (vector): rho=-0.0002, u_g1=1.6119e-02, u_g2=1.3874e-02, score_HM3=1.9657e-02, score_FAM07=4.9681e-10
- `static-cex` b1 `frame_S6_greedy_v2` (vector): rho=-0.0001, u_g1=1.7286e-02, u_g2=1.6799e-02, score_HM3=1.7039e-02, score_FAM07=5.5125e-11
- `static-cex` b12 `frame_S6_greedy_v2` (vector): rho=-0.0001, u_g1=1.6401e-02, u_g2=1.4908e-02, score_HM3=2.0603e-02, score_FAM07=6.0234e-11
- `static-cex` b12 `frame_S6_greedy` (frame): rho=0.0000, u_g1=3.1831e-02, u_g2=2.9723e-02, score_HM3=4.0729e-02, score_FAM07=5.3384e-10
- `static-cex` b2 `frame_S6_greedy` (frame): rho=0.0000, u_g1=3.0448e-02, u_g2=2.9493e-02, score_HM3=3.9786e-02, score_FAM07=2.1815e-09
- `static-cex` b31 `frame_S6_greedy` (frame): rho=0.0000, u_g1=3.0774e-02, u_g2=2.8355e-02, score_HM3=3.9124e-02, score_FAM07=7.2124e-09
- `static-cex` b1 `frame_S6_greedy` (frame): rho=0.0000, u_g1=3.4284e-02, u_g2=3.3505e-02, score_HM3=3.3890e-02, score_FAM07=8.1573e-09
- `mixed-tail-sharp` b31 `frame_S6_greedy` (frame): rho=0.0000, u_g1=2.8745e-02, u_g2=2.9326e-02, score_HM3=3.8709e-02, score_FAM07=3.0992e-07
- `diffuse-diffuse` b12 `frame_S6_greedy` (frame): rho=0.0000, u_g1=2.8182e-02, u_g2=2.8096e-02, score_HM3=3.7539e-02, score_FAM07=3.5540e-07
- `mixed-tail-sharp` b12 `frame_S6_greedy` (frame): rho=0.0000, u_g1=2.9047e-02, u_g2=2.8278e-02, score_HM3=3.8174e-02, score_FAM07=5.5244e-07
- `diffuse-diffuse` b31 `frame_S6_greedy` (frame): rho=0.0000, u_g1=2.8251e-02, u_g2=2.8232e-02, score_HM3=3.7697e-02, score_FAM07=7.5409e-07

Notes:

- Vector `score_FAM07` is `score_HM3 * rho_F^2`; frame `score_FAM07` is `score_HM3 * rho_frame`.
- The signature test excludes oracle rows and compares v1/v2 candidates against the matching projected oracle vector; frame rows compare against `frame_oracle_proj`.
- This is T2 diagnostic evidence only; it does not wire FAM-07 into the streaming score path.
