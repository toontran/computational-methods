# Random Finite-Sample Separation: iSVD vs Stability Scoring

Date: 2026-04-30
Backlog: `summary/overview/score_family_workflow.txt` [TH-05]
Hardens: `summary/theory_isvd_entropy_regime.md` [TH-04]; resolves the
"adversarial-ordering" objection to Q18.

## Verdict

The transient-spike vs recurring-diffuse separation between
streaming-iSVD and a current/validation stability score is *not* an
artifact of adversarial row placement. It survives iid randomization at
non-negligible probability. The Monte Carlo grid in
`random_separation_grid.csv` reproduces

- the closed-form single-spike probability `m p (1-p)^(m+q-1)` to within
  Monte Carlo noise (~1.5e-3 absolute at 2e5 trials);
- a `~9.3 %` joint separation event at the canonical setting
  `m=32, q=96, a=1, L=8, p=1/128`, where streaming iSVD picks the
  transient direction `u`, the current-plus-validation norm picks the
  recurring direction `v`, and HM2(`||A_cur x||^2`,`||A_fut x||^2`) picks
  `v`.

Randomization removes adversarial placement; it does not remove the
finite-sample leverage variance that makes iSVD vulnerable. The
shortcoming surfaced in TH-04 is therefore a property of the streaming
selection rule plus iid sampling, not a property of a hand-built
counterexample matrix.

## Minimal iid Model

Two unit candidate directions `u, v in R^d`. Every row independently:

- response in direction `v`: deterministic `a` (diffuse, every row same);
- response in direction `u`: `L` with probability `p`, else `0` (rare
  spike).

`m` current rows, `q` validation rows, no prior carry (`B_prev = 0`).
With `K_cur = #{current u-spike rows}` and `K_fut = #{validation u-spike
rows}`,

```text
||A_cur u||^2 = K_cur L^2     ||A_cur v||^2 = m a^2
||A_fut u||^2 = K_fut L^2     ||A_fut v||^2 = q a^2
K_cur ~ Bin(m, p)             K_fut ~ Bin(q, p), independent
```

Three selection rules are compared:

```text
iSVD (streaming gain):     pick argmax_x ||A_cur x||^2
stacked (full evidence):   pick argmax_x ||[A_cur; A_fut] x||^2
HM2  (stability score):    pick argmax_x HM2(||A_cur x||^2, ||A_fut x||^2)
                            with HM2(a,b) = 2 a b / (a + b), 0 if a+b=0.
```

## Closed-Form Separation Event

The "one current spike, no validation spike" event is

```text
P(K_cur = 1, K_fut = 0) = m p (1 - p)^{m + q - 1}.
```

Conditional on the parameter window `m a^2 < L^2 < (m + q) a^2`, this
event yields

- iSVD picks `u`:   `||A_cur u||^2 = L^2 > m a^2 = ||A_cur v||^2`;
- stacked picks `v`: `||A_cur u||^2 + ||A_fut u||^2 = L^2 + 0 < (m+q) a^2`;
- HM2 picks `v`:   `HM2_u = 0` (no validation support), `HM2_v > 0`.

The maximum of `m p (1-p)^{m+q-1}` over `p` is at `p* = 1/(m+q)` and
attains `(m/(m+q)) (1 - 1/(m+q))^{m+q-1} -> (m/(m+q)) e^{-1}` as
`m + q -> infty`. So at every window size with `m/q` fixed, the optimal
single-spike probability is bounded below by a constant. There is no
parameter regime in which the separation collapses to negligible
probability — the unfavorable event is at least order `1/e * m/(m+q)`.

For the canonical setting `m=32, q=96`, `p* = 1/128`, and the limiting
probability is `0.25 * e^{-1} ≈ 0.0920`.

## Empirical Grid

Driver: `probe_random_separation.py`. Output: `random_separation_grid.csv`,
`random_separation_grid.json`. 200000 trials per row. Selected rows
below; full grid in CSV.

### Canonical setting and `p` sweep

`m = 32`, `q = 96`, `a = 1`, `L = 8`, so the parameter window is
`32 < L^2 = 64 < 128`. iSVD picks `u` whenever `K_cur >= 1`. Stacked
picks `v` whenever `K_cur + K_fut <= 1`. HM2 picks `v` whenever
`K_fut = 0` (with `K_cur >= 1` already; otherwise both rules tie at
HM2 = 0 trivially).

| label                          |     p     | P_single_theory | freq_single | freq_isvd→u | freq_stack→v | freq_HM2→v | freq_joint |
| ------------------------------ | --------: | --------------: | ----------: | ----------: | -----------: | ---------: | ---------: |
| sweep_p_canonical_p0.00097656  | 1/1024    |        0.027603 |    0.026935 |      0.0302 |       0.9930 |     0.9972 |    0.026935 |
| sweep_p_canonical_p0.0019531   | 1/512     |        0.048758 |    0.049525 |      0.0613 |       0.9738 |     0.9897 |    0.049525 |
| sweep_p_canonical_p0.0039062   | 1/256     |        0.076039 |    0.075985 |      0.1181 |       0.9098 |     0.9626 |    0.075985 |
| **sweep_p_canonical_p0.0078125** | **1/128** |    **0.092331** | **0.092540** |  **0.2218** |   **0.7368** | **0.8826** | **0.092540** |
| sweep_p_canonical_p0.015625    | 1/64      |        0.067665 |    0.068025 |      0.3954 |       0.4034 |     0.6920 |    0.068025 |
| sweep_p_canonical_p0.03125     | 1/32      |        0.017737 |    0.017775 |      0.6398 |       0.0874 |     0.3901 |    0.017775 |
| sweep_p_canonical_p0.0625      | 1/16      |        0.000551 |    0.000590 |      0.8717 |       0.0025 |     0.1302 |    0.000590 |

Reading. The empirical single-spike frequency tracks the closed-form to
~1e-3. At `p* = 1/128` the joint separation event is ~9.3 %; at
`p = 1/512` it is still ~5 %. The marginal numbers tell the rest of the
story:

- iSVD-picks-u rate climbs monotonically with `p` (more spike rows means
  more chances `K_cur >= 1`); at `p = 1/16` iSVD picks the spike 87 % of
  the time.
- Stacked-picks-v rate falls with `p` (more spikes in validation make
  the full-norm winner flip to `u`).
- HM2-picks-v rate is uniformly the highest of the three for `v` and
  degrades the slowest as `p` rises, because the only way HM2 picks `u`
  is for *both* windows to fire spikes.

The joint event maximizes near `p*`, as the calculus predicts.

### `L/a` sweep at `p = 1/128` (canonical `m, q, a`)

Parameter window: `sqrt(32) ≈ 5.657 < L < sqrt(128) ≈ 11.314`.

| L   | L^2 | inside window? | freq_isvd→u | freq_stack→v | freq_HM2→v | freq_joint |
| --: | --: | :-: | --: | --: | --: | --: |
|  4.0|  16 | no  | 0.0021 | 1.0000 | 0.9999 | 0.00198 |
|  5.0|  25 | no  | 0.0259 | 0.9995 | 0.9957 | 0.02153 |
|  5.7|  32.49 | yes (just) | 0.2221 | 0.9814 | 0.9870 | 0.20902 |
|  6.0|  36 | yes | 0.2226 | 0.9818 | 0.9524 | 0.17494 |
|  7.0|  49 | yes | 0.2222 | 0.9199 | 0.8824 | 0.10372 |
|  8.0|  64 | yes | 0.2213 | 0.7357 | 0.8826 | 0.091895 |
|  9.0|  81 | yes | 0.2211 | 0.7365 | 0.8824 | 0.091595 |
| 10.0| 100 | yes | 0.2224 | 0.7346 | 0.8824 | 0.092635 |
| 11.0| 121 | yes | 0.2219 | 0.7375 | 0.8834 | 0.09323  |
| 11.5| 132.25 | no | 0.2220 | 0.3664 | 0.8821 | 0.0    |
| 13.0| 169 | no  | 0.2210 | 0.3675 | 0.8829 | 0.0    |

Reading. Inside the parameter window the joint event holds; just outside
the lower edge (`L = 4, 5`) iSVD almost never picks `u` (the spike's
energy isn't above `m a^2`); just outside the upper edge (`L >= 11.5`)
the validation block can no longer outvote a single spike, so the
stacked rule starts agreeing with iSVD and the joint is empty. The
sharp `L = 5.7` row (just above `sqrt(m a^2)`) gives the largest joint
rate — the spike just clears the iSVD threshold and stacked still
prefers `v` even when `K_cur = 2, K_fut = 0` because `2 L^2 < (m+q) a^2`.

### Window-size scaling at `p = 1/(m+q)` (mid-window `L`)

| (m, q)      | L     | freq_single | freq_isvd→u | freq_stack→v | freq_HM2→v | freq_joint |
| ----------- | ----: | ----------: | ----------: | -----------: | ---------: | ---------: |
| (8, 24)     |  4.47 |    0.09349  |     0.2251  |       0.7360 |     0.8798 |    0.09349 |
| (16, 48)    |  6.32 |    0.09318  |     0.2245  |       0.7350 |     0.8810 |    0.09318 |
| (32, 96)    |  8.94 |    0.09199  |     0.2216  |       0.7361 |     0.8829 |    0.09199 |
| (64, 192)   | 12.65 |    0.09145  |     0.2211  |       0.7357 |     0.8823 |    0.09145 |
| (128, 384)  | 17.89 |    0.09171  |     0.2209  |       0.7362 |     0.8833 |    0.09171 |

Reading. With `m/q` fixed the joint event probability is a window-size
invariant — it is `1/4 e^{-1} ≈ 0.092` in this `m/q = 1/3` family for all
window sizes, just as the closed form predicts. The TH-05 separation is
not a small-window artifact.

### `q/m` ratio sweep (`m = 32`, mid-window `L`, `p = 1/(m+q)`)

| q   | freq_single | freq_isvd→u | freq_stack→v | freq_HM2→v | freq_joint |
| --: | ----------: | ----------: | -----------: | ---------: | ---------: |
|  16 |    0.24945  |     0.4908  |       0.7370 |     0.8604 |    0.24945 |
|  32 |    0.18499  |     0.3946  |       0.7374 |     0.8437 |    0.18499 |
|  64 |    0.12223  |     0.2840  |       0.7358 |     0.8607 |    0.12223 |
|  96 |    0.09213  |     0.2215  |       0.7359 |     0.8829 |    0.09213 |
| 192 |    0.05206  |     0.1333  |       0.7350 |     0.9224 |    0.05206 |
| 384 |    0.02875  |     0.0746  |       0.7346 |     0.9553 |    0.02875 |

Reading. Larger `q/m` (more validation evidence) makes HM2 even more
reliable but lowers the optimal `p* = 1/(m+q)`, which in turn lowers the
unconditional joint rate. The point is the opposite: with very little
validation (`q = 16`), the joint event hits ~25 % — there is *more*
finite-sample selection bias when the validation window is small,
exactly the streaming regime we care about.

## Why iSVD-Picks-u Stays at ~22 % Even at `p*`

A natural worry: 22 % feels high. It is. The reason is that iSVD picks
`u` whenever `K_cur >= 1`, not only on the perfect single-spike event.
At `p = 1/128, m = 32`,

```text
P(K_cur >= 1) = 1 - (1 - p)^m = 1 - (127/128)^32 ≈ 0.224.
```

The full-evidence rule cleans up most of these (it only loses when
`K_cur + K_fut >= 2` aligns badly), and HM2 cleans them up further
because it requires *both* current and validation support to score `u`
above `v`. The 22 % iSVD failure rate is not a bug; it is the
finite-sample leverage variance the alternative scores are designed to
mitigate.

## Conclusion

The TH-04 separation is a real iid finite-sample phenomenon, not a
hand-built ordering. The relevant takeaways for report-level claims:

1. The single-spike event probability is `m p (1 - p)^{m+q-1}`. Its
   maximum over `p` is `(m/(m+q)) e^{-(m+q-1)/(m+q)}`, bounded below by
   `(m/(m+q)) e^{-1}`. There is no parameter regime that makes this
   probability negligible at fixed `m/q`.
2. At the canonical `m=32, q=96, a=1, L=8, p=1/128` setting, iSVD picks
   the transient direction 22 % of the time, the current-plus-validation
   norm picks `v` 74 % of the time, and HM2 picks `v` 88 % of the time.
   The joint TH-04 separation event happens 9.3 % of trials.
3. HM2 dominates iSVD across the iid grid for direction selection
   precisely because it requires evidence in both windows, which a rare
   spike fails to provide. Stacked is intermediate and degrades when `p`
   is too high to keep the validation block clean.
4. Outside the parameter window `m a^2 < L^2 < (m+q) a^2` the
   separation collapses, as expected. Inside, it is robust to all three
   sweeps (`p`, `L/a`, window size).
5. *Randomization removes adversarial placement; it does not remove
   finite-sample leverage variance.* iSVD's vulnerability is not a
   pathology of the test matrices in §6 — it is a generic property of
   maximizing observed stacked norm without window-stability evidence.

This closes Q18 in its full hardened form: TH-04 supplies the
deterministic example, TH-05 supplies the iid robustness check, and
neither of them claims "iSVD is wrong everywhere" — only that
streaming-iSVD's selection rule is biased toward transient leverage in
the high-sample-entropy finite-window regime, which is exactly where the
score family is meant to be the better tool. The low-sample-entropy
regime (where one window genuinely contains most of the signal) remains
iSVD's. The complementarity story stands.

## Files

- `probe_random_separation.py` — driver script (working dir).
- `summary/theory_isvd_random_finite_sample/random_separation_grid.csv`
- `summary/theory_isvd_random_finite_sample/random_separation_grid.json`
- `summary/theory_isvd_random_finite_sample/theory_isvd_random_finite_sample.md`
  (this file).
