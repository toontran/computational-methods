# S7 development context — debugging + next steps

Date: 2026-05-02
Branch: new_orders_evals
Audience: someone picking up the S7-family work next session.

## What S7/S8 are

Variants in `r_sk_g_score.py:r_sk_g_value_grad`:

- **S7**: `score(v) = (raw_sk + raw_g1 + raw_g2) · relH1_stacked`
  - raw_sk = ‖A_sk v‖², raw_g1 = ‖A_cur v‖², raw_g2 = ‖A_fut v‖²
  - relH1_stacked = normalized Shannon entropy of row energies of `[A_sk; A_cur; A_fut] · v`
  - Block-1 fall-through (no sketch): `M_full = [A_cur; A_fut]`.
  - Same form as the original `combined` score, but with A_fut included symmetrically.
- **S8**: `score(v) = raw_sk + raw_g1 + raw_g2`. Same setup as S7 but no entropy term.
- Both restrict the **search basis** to `rowspace([A_sketch; A_cur])` (excludes A_fut from the search domain — the score consumes A_fut as evidence only).

FD gradient check (T1) passes both at rel_err ~1e-10 across b1/b2/b12/b31. Source of truth: `r_sk_g_score.py --gradient-check`.

## Window-asymmetry parameter

CLI knobs in `half_window_sliding_hmean_experiment.py`:

- `--h1-mult` / `--h2-mult`: multipliers on `--half-win`. Default 1.0/1.0 (symmetric, current §6 §6 setting).
- `h1 = max(1, round(half_win * h1_mult))`, `h2 = max(0, round(half_win * h2_mult))`.
- `step = h1 if sliding else h1 + h2`.
- Loop bounds: `for start0 in range(0, max(0, n − h1 − h2 + 1), step)`. Allows h2=0.

Each round commits h1 rows to the sketch; h2 is the peek window (used by the score, not committed). With h1=2.0·half_win and h2 small, you get a "combined-like" wide-search-no-peek setup.

## Wiring difference: combined vs S7 (CRITICAL)

In `run_pair_stream` (lines ~786-820):

- **combined policy**: V_selected = V_default = joint rank-2 from `entropy_iter_basis_forget` with `score_variant="combined"`. slot-2 = V_default[:,1].
- **future_hmean_r_sk_g policy** (S7/S8): V_default[:,0] inherited from combined (same). chosen_v2 = output of S7 gradient ascent in deflated complement of V_default[:,0]. Then **V_default[:,1] is discarded** and V_selected = `rank2_svd_frame(V_default[:,0], chosen_v2_S7, M_gain)` — top-2 SVD of M_gain restricted to span(V_default[:,0], chosen_v2_S7).

So at the V level: combined uses joint rank-2; S7 uses combined's slot-1 + greedy-deflated-S7 slot-2 + svd_frame substitution.

This wiring is the dominant reason original-S7 underperforms combined on matrices like diffuse-diffuse / mts-soft despite seeing more evidence (peek + stacked entropy).

## Pareto override (`--rsk-pareto-metric`)

Implemented in `half_window_sliding_hmean_experiment.py`:

- `off`: original behavior (chosen_v2_S7 always replaces V_default[:,1]).
- `s7frame`: pick by S7 frame score `(||A_sk V||_F² + ||A_cur V||_F² + ||A_fut V||_F²) · relH(stacked)`. Helps mts-sharp slightly; leaves diffuse-diffuse unchanged because S7's score genuinely prefers chosen_v2_S7 over V_default[:,1] there.
- `mgain`: pick by `||M_gain V||_F²` (combined's objective). **This is the strong variant**. Wins:
  - diffuse-diffuse 0.483/0.248 (combined) → 0.708/0.242 (Pareto-mgain S7)
  - mts-balanced 0.756/0.075 → 0.910/0.605
  - mts-soft 0.886/0.304 → 0.903/0.755 (the formerly-regression matrix)
  - static-cex tied within noise
- `always-default`: V_selected = V_default unchanged. Sanity control: gives exact match to combined.

Single seed = 0, half_win = 32. Cross-seed verification not done.

`s7_frame_score` and the override are in `half_window_sliding_hmean_experiment.py` lines ~279 and ~786-830. CLI flag added near line ~1100.

## Combined with h1=64 (= S7 2.0/~0 score domain, combined optimizer)

Running combined with `--h1-mult 2.0 --h2-mult 0.03125` (h2=1) gives:

| matrix | combined h1=32 | combined h1=64,h2=1 | combined h1=64,**h2=0** |
|---|---|---|---|
| diffuse-diffuse | 0.483/0.248 | 0.885/0.002 | **0.894/0.778** |
| mts-sharp | 0.894/0.019 | 0.845/0.009 | 0.854/0.014 |
| mts-balanced | 0.756/0.075 | 0.865/0.044 | 0.906/0.036 |
| mts-soft | 0.886/0.304 | 0.896/0.166 | 0.919/0.295 |
| static-cex | 0.975/0.912 | 0.963/0.961 | 0.994/0.044 |

**KEY ARTIFACT TO BE AWARE OF**: h2=0 vs h2=1 changes round count by 1 (h2=0 gets 16 rounds = all 1024 rows committed to sketch; h2=1 gets 15 rounds = last 64 rows skipped). On diffuse-diffuse and static-cex this single extra round flips slot-2 cos1² by 0.7+. **Round count, not score, drives most of these numbers.**

Implication: any cross-h1/h2 comparison must be locked to identical round count via `--max-pairs` to be apples-to-apples. The §6 ladder I produced (1.0/1.0 vs 1.5/0.5 vs 0.5/1.5) has uneven round counts and is not strictly fair. **Re-run with locked round count before drawing strong conclusions about window asymmetry.**

## Optimizer round-count sensitivity

Comparing combined h1=64 (16 rounds, entropy_iter Krylov optimizer) vs S7 hw=32 2.0/~0 (16 rounds, gradient-ascent + 24 random restarts, essentially equivalent score when h2≈0):

- diffuse-diffuse cos1²: combined h1=64,h2=1 gives 0.002; S7 hw=32,2.0/~0 gives 0.311.
- Same effective score, same round count. Different optimizers.

Combined's entropy_iter is **round-count sensitive**: it needs many (~32) joint rounds to lock slot-2. S7's gradient ascent + 24 restarts is more robust per-block but lacks the iterative refinement structure.

This is a **real handle to pull on** for further design. A combined-style score with S7-style optimizer (gradient ascent + many restarts) at h1=64 might combine the best of both.

## Files (where to look)

- Code:
  - `r_sk_g_score.py` — S1–S8 score variants, FD check.
  - `half_window_sliding_hmean_experiment.py` — bench harness, streaming loop, Pareto override, h1/h2 CLI flags.
- Synthesis docs (read these first):
  - `summary/score_family_fullsum_pastcurrent_search/synthesis.md` — initial S7/S8 and 2-matrix sanity.
  - `summary/score_family_asym_window/synthesis.md` — asymmetric window ladder.
  - `summary/score_family_pareto_override/synthesis.md` — mgain Pareto results.
  - `summary/score_family_combined_h1_64/` and `summary/score_family_combined_h1_64_h2_0/` — combined large-h1 results.
- Probes:
  - `probe_s7_peek_compare.py` — per-block diagnostic on static-cex (with-peek vs no-peek slot-2, cos² vs oracles, score components, residual analysis). Used to find the structural raw_g2 ≈ 0 cause and the v=I oracle visibility issue.

## Open questions / next steps

1. **Round-count fairness.** Re-run the asymmetric window ladder with `--max-pairs N` locked to a common N (e.g. 14) so 1.0/1.0, 1.5/0.5, 0.5/1.5, 2.0/0 all process identical row count. This is the FIRST priority before any further comparison.

2. **Cross-seed robustness of Pareto-mgain.** The mts-soft 0.755 cos1² and dd 0.708/0.242 are single-seed (=0). Run seeds {0..4} on the 5 priority matrices to estimate seed variance. If 0.45 cos1² lift on mts-soft holds, this is a real result.

3. **Combined-score + S7-optimizer hybrid.** Implement a new policy that uses combined's score formulation (`(||Bv||² + ||A_cur v||²) · phi`) but with S7's gradient-ascent + random-restart optimizer, at h1=64 / h2=0. Specifically targets the "combined optimizer is round-count sensitive" finding — if S7's optimizer recovers slot-2 with combined's score at 16 rounds, it isolates score (fine) from optimizer (the bottleneck).

4. **A clean fork.** User wanted a fork of combined as a starting point. The minimal fork would add a new policy (e.g. `combined_fork`) that copies combined's wiring, then can be modified independently. Step 1 of the fork must reproduce `combined h1=X` numbers exactly. Then add modifications (e.g. include A_fut in score, or change optimizer).

5. **Joint Stiefel-2 with S7 score.** The current S7 wiring uses greedy deflation. A joint Stiefel-2 optimizer on S7's frame score would likely match combined's slot-2 quality. FAM-01 already showed rank-r joint with HM3 evidence FAILS-subspace on diffuse-diffuse — but S7's score (sum × stacked entropy, not HM3) is different, so worth trying.

6. **Why does mgain Pareto win mts-soft so big (+0.45 cos1²)?** Combined's joint optimizer leaves a high-M_gain-energy direction on the table that S7's optimizer finds. Worth a focused per-block diagnostic to characterize this direction (is it oracle-aligned? where is it in V_default's plane?) and understand whether it's exploiting an optimizer flaw or genuine score complementarity.

7. **DOC-01 / CLAUDE.md hygiene.** When this work consolidates, update `summary/overview/score_design_overview.txt` to add S7/S8/Pareto-override as siblings of S1-S6/D0/S6_OP/S6_E2 in the score family.

## Acceptance criteria for the next ship

Before claiming a §6 ship target met:

- All §6 7-matrix bench numbers at `--half-win 32 --row-shuffle-seed 0`.
- Locked round count across compared variants.
- Pass §1quinquies S-1 (vector oracle gap) and S-2 (frame oracle gap, two-part rule with τ_align=0.5 for slot-r ship).
- Cross-seed verification on at least seeds {0, 1, 2}.

The §6 baseline §6 high-entropy regime no-regression criterion (don't drop below S6 ref on the 4 high-entropy matrices) still applies. Pareto-mgain currently has a small mts-sharp cos0² regression (-0.056) that needs cross-seed validation.

## Known-broken expectations to NOT repeat

- "Window asymmetry strictly orders results" — not at the cos1² level once round count is locked. Round count dominates.
- "More peek info → strictly better S7" — false. The greedy-vs-joint slot-2 wiring matters more than peek info.
- "Combined optimizer behaves identically at any window size" — false. Round-count-sensitive. Slot-2 needs ≥32 rounds on diffuse-diffuse.
- "rowspace(A_cur) ⊥ rowspace(A_fut) is exact on static-cex" — only ~5e-5 in cos², not exact. Driven by the U-orthogonalization tail leakage; the +0.031 col-0 entries DO carry small overlap that gets cancelled by the −0.0019 cross-block tail terms.
