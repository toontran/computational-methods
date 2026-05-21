# Asymmetric window split for S7 (full-sum × stacked entropy)

Date: 2026-05-01

## Knob

`--h1-mult` and `--h2-mult` in `half_window_sliding_hmean_experiment.py`
parameterize the streaming loop's two halves:

- `h1 = round(half_win * h1_mult)` — first half (search/commit, sketch absorbs).
- `h2 = round(half_win * h2_mult)` — second half (peek; rolls into next round).
- `step = h1 if sliding else h1 + h2`. Sketch advances by h1 each round.

Default 1.0/1.0 reproduces existing behavior. Tested 1.5/0.5 and 0.5/1.5
against symmetric S7 on the priority §6 subset (diffuse-diffuse,
mixed-tail-{sharp,balanced,soft}, static-cex).

## Results (cos0² / cos1², half_win=32, seed=0)

| matrix              | S6 ref (§6) | S7 1.0/1.0    | S7 1.5/0.5      | S7 0.5/1.5    |
|---------------------|-------------|---------------|-----------------|---------------|
| diffuse-diffuse     | 0.298/0.131 | 0.595/0.040   | **0.867/0.142** | 0.752/0.035   |
| mixed-tail-sharp    | 0.861/0.111 | 0.778/0.028   | **0.899/0.263** | 0.962/0.002   |
| mixed-tail-balanced | 0.750/0.018 | **0.907/0.111** | 0.788/0.009   | 0.754/0.003   |
| mixed-tail-soft     | 0.941/0.156 | 0.890/0.002   | 0.808/0.050     | 0.810/0.002   |
| static-cex          | 0.972/0.154 | 0.983/0.019   | **0.981/0.273** | 0.978/0.000   |

Bold = best of the three S7 splits per row.

## Pareto wins vs S6 §6 baseline

- **diffuse-diffuse, 1.5/0.5**: cos0² +0.569, cos1² +0.011
- **mixed-tail-sharp, 1.5/0.5**: cos0² +0.038, cos1² +0.152
- **mixed-tail-balanced, 1.0/1.0**: cos0² +0.157, cos1² +0.093
- **static-cex, 1.5/0.5**: cos0² +0.009, cos1² +0.119

## Observations

1. **1.5/0.5 wins slot-2 on 3 of 5 priority matrices** (diffuse-diffuse,
   mts-sharp, static-cex). The bigger commit basis (h1=48 rows + sketch) lets
   the search find oracle-aligned slot-2 directions that were unreachable
   with h1=32; small h2=16 still gives enough between-window evidence.
2. **0.5/1.5 buys cos0² but kills cos1²** consistently. Small commit basis
   (h1=16) cannot separate two oracle directions in the deflated complement;
   slot-2 collapses (cos1² ≤ 0.04 on every matrix). The bigger peek improves
   slot-1 stability evidence (cos0² up on 2 matrices) but the score cannot
   resolve the slot-2 picker.
3. **Symmetric 1.0/1.0 still wins on mts-balanced**: it's a Pareto-balanced
   regime where the asymmetry doesn't help.
4. **mixed-tail-soft is the only regression**: every S7 variant underperforms
   S6's 0.941/0.156. This matrix wants something different from raw-sum ×
   stacked-entropy; not addressable via window asymmetry alone.

## Code touch points

- `half_window_sliding_hmean_experiment.py:434-440`: loop now uses h1/h2.
- `half_window_sliding_hmean_experiment.py:1100-1106`: new `--h1-mult` and
  `--h2-mult` CLI args, default 1.0.
- No score-function changes (S7/S8 unchanged from
  `summary/score_family_fullsum_pastcurrent_search/synthesis.md`).

## 2026-05-01 extension — combined and S7 2.0/~0 ladder

Full parametric ladder (cos0²/cos1², same harness):

| matrix              | combined    | S7 2.0/~0   | S7 1.5/0.5  | S7 1.0/1.0  | S7 0.5/1.5  |
|---------------------|-------------|-------------|-------------|-------------|-------------|
| diffuse-diffuse     | 0.483/0.248 | **0.889/0.311** | 0.867/0.142 | 0.595/0.040 | 0.752/0.035 |
| mixed-tail-sharp    | 0.894/0.019 | 0.780/0.300 | **0.899/0.263** | 0.778/0.028 | 0.962/0.002 |
| mixed-tail-balanced | 0.756/0.075 | 0.855/0.110 | 0.788/0.009 | **0.907/0.111** | 0.754/0.003 |
| mixed-tail-soft     | 0.886/**0.304** | 0.893/0.257 | 0.808/0.050 | 0.890/0.002 | 0.810/0.002 |
| static-cex          | 0.975/**0.912** | 0.967/0.310 | 0.981/0.273 | 0.983/0.019 | 0.978/0.000 |

S7 2.0/~0 means h1=64, h2=1 (h2_mult=0.03125 to keep h2≥1).

Findings:
- **S7 2.0/~0 substantially beats combined on diffuse-diffuse** (cos0²
  +0.406, cos1² +0.063). Combined's M3 (blind to A_fut) shows up
  here; even an entropy bias on [sketch; A_h1] alone (peek=1 row) outperforms
  combined's phi on the same domain.
- **No single split dominates**: diffuse-diffuse → 2.0/~0; mts-sharp →
  1.5/0.5; mts-balanced → 1.0/1.0; mts-soft ≈ tie combined; static-cex →
  combined-greedy unchallenged. Suggests `(h1_mult, h2_mult)` as a
  per-matrix or adaptive parameter.

## Open questions

1. Does 1.5/0.5 generalize to other seeds, and to S8 (no entropy)?
2. Are there matrix-specific optima (e.g. 1.25/0.75, 1.75/0.25)?
3. The mts-soft regression: is this a fundamental S7-evidence-model
   limitation, or just a single-seed unlucky run?
4. Does the asymmetric split help S6 or only the new S7 family?
5. Can the optimal split be picked online from a value-only signal (e.g.
   carry maturity, current-vs-future variance ratio)?

## Files

- `summary/score_family_asym_window/{matrix}_S7_h1-{m1}_h2-{m2}.{json,csv,txt}`
  for matrix ∈ {diffuse-diffuse, mixed-tail-{sharp,balanced,soft},
  static-cex} (mts/sc 1.0/1.0 from
  `summary/score_family_fullsum_pastcurrent_search/`).
