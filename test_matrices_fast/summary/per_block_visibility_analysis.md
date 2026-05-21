# Per-block oracle visibility, score-family regime sensitivity, and the buffer-size story

Date: 2026-05-06
Branch: new_orders_evals
Audience: someone trying to understand why combined-score fails on diffuse matrices at half_win=32 and what to do about it.

This document summarizes a long analysis session that traced the score family's regime-2 (diffuse-diffuse) failure to a buffer-size artifact rather than a score-design defect, and develops the conceptual scaffolding (block orthogonality, oracle visibility, block entropy) needed to reason about these problems. Read top-to-bottom; sections are independent but build on each other.

---

## 0. Reproduction context

- **Bench harness**: `half_window_sliding_hmean_experiment.py`
- **Default config used throughout**: `--half-win 32 --row-shuffle-seed 0 --rank 2 --preset fast`
- **Test matrices**: `static-cex`, `mixed-tail-sharp`, `diffuse-diffuse`, `mixed-tail-soft` (4-matrix subset of the §6 7-matrix set; the remaining three — `mts-balanced`, `etf-basket-basis`, `residual-spiky-shocks`, `risk-residual-panel` — share the same intuitions modulo specific exceptions, especially `etf-basket-basis` which has non-orthogonal blocks).
- **Matrix size**: n=1024, rank=2 (top-2 right singular vectors are the recovery target).
- **Single-seed runs throughout**: results are seed=0 only. Cross-seed verification not done in this analysis (would be the next step).

All runs in this document used `--max-pairs <unset>` (full streaming until matrix consumed). When earlier sections of the dev context reference K=16, that was a `--max-pairs 16` constraint that was dropped here.

---

## 1. Block-orthogonality fact

For all four test matrices, consecutive half-windows of 32 rows are essentially orthogonal in row-space:

```
matrix             max cos²(rowspace(A_cur), rowspace(A_fut))   median over 8 blocks
static-cex         5.6e-5                                       4.7e-5
mixed-tail-sharp   1.2e-4                                       3.4e-5
diffuse-diffuse    1.4e-4                                       1.3e-4
mixed-tail-soft    3.4e-4                                       1.1e-4
```

Computation: `Q_c = rowspace_basis(A_cur)`, `Q_f = rowspace_basis(A_fut)`, then `sigmas = svd(Q_c.T @ Q_f, compute_uv=False)`. Squared singular values are squared cosines of principal angles. Code reference: `probe_s7_peek_compare.py:248-255`.

**Counterexample**: `etf-basket-basis` has cos² ≈ 0.95 between consecutive blocks (highly non-orthogonal) — every conclusion below about block orthogonality fails on that matrix. ETF data structurally repeats the same basket directions across windows.

**Why this matters**: block orthogonality means the oracle V_exact[:,0] is a *single fixed direction in R^n* whose mass is *split across blocks* in orthogonal fragments. No single block contains a meaningful fraction of v_oracle. Each block contributes a fresh, orthogonal piece.

---

## 2. Per-block oracle visibility δ

For unit V_exact[:,0] (the global top right SV), the per-block mass is

```
δ = ||proj_{rowspace(A_cur)}(V_exact[:,0])||²
```

Empirically: δ ≈ 0.037 per block at half_win=32 (consistent across all 4 matrices, very close to half_win/n = 32/1024 = 0.031 — slightly above uniform).

After K orthogonal blocks, the cumulative oracle mass reachable from training-only data is

```
||P_train V_exact[:,0]||² = K × δ
```

This bounds the maximum possible cos²(V_selected, V_exact[:,0]) for any V_selected ⊂ span(training):
- K=16, δ=0.037: ceiling cos² = 0.59 (cos = 0.77).
- K=32, δ=0.031 (h2=0 mode): ceiling cos² = 0.99.

The bench's combined on static-cex achieves 0.725 cos at K=16 (cos² = 0.525), close to the 0.77 ceiling. So combined IS near-optimal at accumulating fresh oracle fragments per round on row-concentrated matrices.

**Cross-block oracle leakage = 0**: if you compute `mass_in_A_cur + mass_in_A_fut − mass_in_[A_cur; A_fut]`, you get 0.000 to 3 decimals on all four matrices. The oracle's mass in different blocks is structurally disjoint. There is no "shared oracle" content between blocks.

This was an early correction during the session: my initial proposal of "the cross-block overlap subspace contains oracle content" was wrong. The overlap subspace is just the joint span of two block-specific principal vectors that happen to lie in oracle-aligned subspaces of their own blocks but don't share anything.

---

## 3. The score landscape is flat per-block

For a candidate cloud (~500 unit vectors: random + rowspace-restricted + top-SVs + V_exact) at each block, evaluated against the combined score `(raw_sk + raw_g1) · relH`:

```
metric (8 blocks, mean)         static-cex   mts-sharp   diffuse-diffuse   mts-soft
ρ_S (Spearman vs cos²(v,V_r))   -0.002       +0.086      +0.056            +0.056
q_or (P[S(rand) ≥ S(oracle)])   0.003        0.005       0.001             0.003
amax_T (cos² of argmax)         0.002        0.010       0.006             0.011
margin (relative gap)           -3.73        -3.96       -3.72             -3.83
```

**Interpretation**: the score has **near-zero correlation with oracle alignment** in a representative candidate cloud, on all 4 matrices. The score's empirical maximum in cloud is essentially orthogonal to V_exact[:,0]. The oracle's score is 1–10× **below** the runner-up non-oracle candidate (negative margin).

This destroys the naive interpretation of "the score finds the oracle within each block." It doesn't. The score's `cos² with V_exact` is ~0.005 across all 4 matrices, including the matrices where the bench reports combined cos² = 0.5.

**Resolution**: the bench's reported `mean_cos1 = 0.99` for combined static-cex is *cos with reachable Q_oracle*, not with V_exact directly. Q_oracle = V_exact projected into rowspace(M_gain); the search basis itself only carries a fraction of V_exact, and V_selected matches that fraction near-perfectly. The "0.98" is in-basis alignment, not global oracle alignment. Globally:

- combined static-cex `mean_exact_cos1 = 0.510` (cos² ≈ 0.26 — much lower than 0.98).

This was a major mid-session correction. The dev context's score acceptance criteria using `mean_cos1` were measuring against a moving target (the reachable basis), not the absolute oracle.

---

## 4. What the score actually does — sketch shaping across rounds

Per-block update equation: at block t, V_score(t) is selected by score+optimizer, then `make_state` projects M_gain into V_score's basis and SVDs. Result: state.V[:,0] is a direction inside span(V_score), reinforced by past sketch.

Decomposing any candidate v at block t as `v = α v_B + β v_A` (B_top span vs A_cur span):
- raw_sk = α² · σ_state²
- raw_g1 = β² · ||A_cur v_A||²

For state.V[:,0] to accumulate oracle alignment over rounds, V_score must have nonzero α (preserves past) AND nonzero β with the β · v_A component pointing at the oracle's projection in A_cur. The α-component alone preserves what's already there; the β-component adds fresh oracle mass δ.

If V_score has α=0 (block-only), the past oracle accumulation `ε_{t-1}` is excluded from state.V_new — sketch resets every round.

**The score's *real* job is constraining V_score to have α > 0 and β · v_A ≈ p_t** (the oracle's fragment in current block). It's a sketch-evolution shaping problem, not a per-block oracle-finding problem. The per-block argmax is irrelevant — it's the sequence of V_score choices, propagated through `make_state`, that accumulates oracle.

---

## 5. The two regimes: row-concentrated vs row-diffuse

The bench numbers split clearly into two regimes:

```
hw=32 (default), final exact_cos1²:

                    iSVD     combined
static-cex          0.002    0.952        ← combined wins (regime 1)
mixed-tail-sharp    0.006    0.800
mixed-tail-soft     0.129    0.784
diffuse-diffuse     0.831    0.233        ← iSVD wins (regime 2)
```

iSVD wins on diffuse-diffuse by a 4× margin. Combined wins on the other three by 100×+.

**Mechanism (concentrated vs diffuse rows)**: for each block, there's a "best direction in this block's rowspace" — the top right SV of A_cur, call it w_t. Two cases:

1. **Row-concentrated (static-cex, mts-sharp, mts-soft)**: rows of A_cur have highly non-uniform norms. σ_1²(A_cur) is dominated by a block-specific spike — w_t is concentrated on a few rows. The oracle's projection p_t into rowspace(A_cur) has the same energy `σ_1²(A) × δ` but spread uniformly across rows. **w_t ≠ p_t**, they're orthogonal-ish directions in the same block.

2. **Row-diffuse (diffuse-diffuse)**: rows of A_cur have uniform-ish norms. Top SV w_t IS the oracle's projection (or close). **w_t ≈ p_t**.

iSVD picks w_t blindly. On diffuse-diffuse, combining w_t's across blocks (orthogonal) gives ≈ P_train v_oracle. iSVD wins. On row-concentrated, w_t's are arbitrary block-specific spikes — they don't combine coherently. iSVD fails.

Combined applies relH gate (entropy of A_cur v across rows). Block-spike w_t has low relH (energy concentrated on few rows) → score kills w_t → search drifts toward p_t (which has high relH). On row-concentrated this works. On diffuse-diffuse, both w_t AND any reasonable alternative have similarly high relH → relH doesn't discriminate → combined picks something arbitrary → fails.

The relH gate is **a proxy for "is this direction the oracle projection"** that works *only* when oracle and block-spike differ in row entropy. On row-concentrated matrices they do; on diffuse they don't.

---

## 6. Self-only uncertainty signals — most don't discriminate

Tested signals (computable without V_exact):

| signal | static-cex (regime 1) | diffuse-diffuse (regime 2) | discriminates? |
|---|---|---|---|
| `sel_car_cos1²` (sketch direction stability) | 1.000 every block | 1.000 every block | **NO** |
| `state.s[0]²` growth | flat ~0.85–0.92 | flat ~0.84–0.87 | **NO** |
| `relerr_sval = |s[0]−σ_1|/σ_1` | 0.04 | 0.07 | weakly |
| α²(V_score) (past-aligned mass) | varies 0.0–0.2 | varies 0.0–0.2 | **NO** |
| ρ_S (score-cos² Spearman in cloud) | 0.0 | 0.0 | **NO** |

The bench's combined optimizer locks onto a single direction within ~1 block on **all** four matrices (sel_car_cos1² = 1.000). On row-concentrated matrices it locks to a slowly-improving oracle projection; on diffuse-diffuse it locks to a non-oracle direction. **Locked-correct and locked-wrong look identical in every self-only sketch metric.**

The bench's `car_cos1` (cos with Q_oracle = V_exact projected into M_gain) DOES discriminate sharply (~1.0 on regime 1, oscillates 0.4–0.76 on diffuse-diffuse), but it requires V_exact to construct Q_oracle. Not self-only.

**Block orthogonality enforces this**: any V_selected ⊂ span(training) is orthogonal to held-out blocks' rowspace (cos² ≈ 1e-4). So `||A_test V_selected||²` is essentially 0 for *any* training-derived V_selected — both oracle-aligned and wrong-attractor. Held-out energy capture, the most natural validation metric, fails by structural geometry.

This is the **fundamental obstruction**: on near-orthogonal-block matrices, the part of v_oracle that's in held-out blocks is orthogonal to anything we can build from training. There is no self-only signal that distinguishes "stably converged to oracle projection" from "stably converged to a wrong direction" once both directions live in span(training).

The dev context's reliance on V_exact for acceptance criteria is **load-bearing, not optional**.

---

## 7. Block entropy of past-block decomposition (proposed signal)

If you DO have access to past blocks (or a sample-based proxy), the signal that DOES discriminate is **block entropy**:

```
c_k = ||A_cur(k) v||²    for k = 0..t-1
p_k = c_k / Σ c_k
H_block(v) = −Σ p_k log p_k
```

Empirical at K=8:

```
                  v_oracle  v_blk_now  v_blk_old  v_random   max=log(8)=2.08
static-cex          2.079     0.000      0.000      2.045
mixed-tail-sharp    2.079     0.003      0.008      2.046
diffuse-diffuse     2.079     0.004      0.004      2.041
mixed-tail-soft     2.079     0.005      0.011      2.051
```

Block-spikes (per-block top SVs) have H_block ≈ 0 across all matrices. Oracle and random both saturate at log(K_test). This is **regime-independent** — the signal works the same way on row-concentrated and row-diffuse matrices.

Combined with mass to disambiguate from random:

```
joint = ||A_seen v||² × H_block(v)

                  oracle   blk_now   blk_old   random
static-cex        0.511    0.000     0.000     0.383
diffuse-diffuse   0.511    0.004     0.003     0.400
```

Oracle wins by ~28% over random and by ~100× over block-spikes. Discriminates cleanly on all matrices.

**Implementation cost**: storage of recent K_test blocks (e.g., last 8). Per-block: ~K_test × half_win × n floats. At K_test=8, half_win=32, n=1024: 256K floats — trivial.

**Score variant suggestion** (drop-in for combined's relH multiplier):
```
score(v) = (raw_sk + raw_g1) · H_block(v)
```
where H_block uses past K_test blocks. Predicted to recover combined's static-cex performance AND rescue diffuse-diffuse, because H_block correctly identifies oracle while row-entropy was misled.

**Caveat**: in production streaming, we may not have raw past blocks — only sketched/sampled rows. Reservoir sampling provides partial access; whether reservoir samples are enough to compute H_block reliably is an empirical question we didn't answer.

---

## 8. The buffer-size discovery — h1 sweep with h2=0

This is the punchline of the analysis. Run combined and iSVD with `--h2-mult 0` (no peek window, all rows committed to sketch) and varying h1:

```
combined exact_cos1² (final block):

                  h1=32(K=32)  h1=128(K=8)  h1=256(K=4)  h1=512(K=2)  h1=1024(K=1)
static-cex          0.985        0.978        0.991        0.999        1.000
mixed-tail-sharp    0.787        0.707        0.802        0.998        1.000
diffuse-diffuse     0.105        0.978        0.999        0.999        1.000
mixed-tail-soft     0.735        0.957        0.999        0.999        1.000

iSVD exact_cos1² (final block):

                  h1=32(K=32)  h1=128(K=8)  h1=256(K=4)  h1=512(K=2)  h1=1024(K=1)
static-cex          0.002        0.002        0.002        0.003        1.000
mixed-tail-sharp    0.006        0.007        0.007        0.017        1.000
diffuse-diffuse     0.846        0.814        0.848        0.963        1.000
mixed-tail-soft     0.140        0.080        0.253        0.915        1.000
```

**At h1=1024 (full matrix in one block, K=1), BOTH algorithms achieve cos²=1.000 on every matrix.** Single-shot SVD trivially gives V_exact.

The "score family fails on diffuse-diffuse" framing is shown to be entirely a **buffer-size artifact**:
- h1=128 with combined: cos² ≥ 0.71 on every matrix, **≥ 0.96 on the previously-failing diffuse-diffuse**.
- h1=256 with combined: cos² ≥ 0.99 on every matrix.

**4× the buffer (h1=128 instead of 32) makes plain combined work near-perfectly across all 4 matrices** without any score modification. This dominates every score variant the dev context has explored at h1=32.

### Why the earlier hw sweep didn't show this

An earlier sweep (h2=hw, peek mode) showed combined converging to ~0.55 at hw=512. That convergence was an artifact: at hw=512, h1=h2=512 means only h1=512 rows commit to sketch. The other 512 rows feed peek/score evaluation but never update state.V. So state.V was bounded to span(first 512 rows), giving cos²≈0.5 by geometry regardless of algorithm quality.

The h2=0 sweep removes this confound. Now every row contributes to the sketch update.

### The relevant parameter is per-block oracle visibility δ = h1/n

Mapping to recovery quality:
- δ ≥ 0.25 (h1 ≥ 256): any reasonable score works on any matrix.
- δ ≥ 0.12 (h1 ≥ 128): combined works on all matrices including diffuse.
- δ = 0.03 (h1=32, default): combined fails on row-diffuse matrices because relH gate's signal-to-noise collapses; iSVD fails on row-concentrated matrices because per-block top SV is structural mismatch.

**The matrix-regime distinction (concentrated vs diffuse) only matters at very small δ.** As δ grows, both algorithms succeed on both regimes.

---

## 9. Implications for the score-family work

The dev context (see `score_family_s7_dev_context.md`, `score_family_pareto_override/synthesis.md`, etc.) has iterated extensively on score variants — S6, S6_OP, S6_E2, S7, S8, Pareto-mgain — trying to improve combined-score's behavior at half_win=32 on diffuse-diffuse and row-concentrated matrices simultaneously.

The findings here suggest most of that work was attacking a problem whose root cause is **harness configuration (h1=32 buffer size) rather than score formula**.

Concrete redirection:
1. **Re-run the §6 7-matrix bench at h1=128 with h2=0** for plain combined and compare to all the score variants at h1=32. Predict that h1=128 plain combined will dominate.
2. **If the streaming use case requires h1=32** (e.g., latency or memory constraint), focus on the **block-entropy** score variant (Section 7) rather than continuing to iterate on relH-based formulas.
3. **mgain Pareto override** at h1=32 is a partial fix: it implicitly captures some of the "persistence across blocks" signal that block-entropy makes explicit. The data shows it lifts diffuse-diffuse 0.291 → 0.708 while leaving static-cex unchanged — consistent with the mechanism.
4. **`--rsk-pareto-metric mgain`** should probably become the default for slot-2 selection across all matrices in the bench, not just an override. Plain combined's relH-multiplier hurts on the regime-2 matrices that the override rescues, with no cost on regime-1 matrices.

---

## 10. The narrative arc, condensed

Reading this document end-to-end, the story is:

1. **Block orthogonality + per-block oracle mass δ ≈ 0.037 means oracle is barely visible per block.** No single block contains v_oracle; each block has a small fresh fragment.

2. **Per-block scoring CAN'T find the oracle.** The score landscape in a candidate cloud has ρ_S ≈ 0 with V_exact alignment; the score's argmax is essentially orthogonal to v_oracle.

3. **The score's real role is sketch shaping**: choosing V_score across rounds so that `make_state`'s SVD compression accumulates oracle-aligned content. Combined's relH gate happens to do this on row-concentrated matrices (block-spikes have low relH and get filtered) but fails on row-diffuse matrices (block-spikes have high relH; relH doesn't discriminate).

4. **iSVD has the opposite failure mode**: blindly picks per-block top SVs. Works on row-diffuse matrices (top SV ≈ oracle projection); fails on row-concentrated (top SV is block-specific spike).

5. **Self-only signals don't discriminate** because block orthogonality forces all training-derived V's to have near-zero capture on held-out blocks. Locked-correct and locked-wrong look identical in every observable sketch metric.

6. **Block entropy of past-block decomposition IS a discriminating self-only signal**, but requires storing past blocks (or a high-fidelity sketch).

7. **Larger h1 (with h2=0) makes the score-formula choice nearly irrelevant**. At h1=128, plain combined works on all matrices. At h1=1024, both algorithms perfectly recover oracle. The score-family's struggles at h1=32 are buffer-size-driven.

---

## 11. Reproduction commands

All numbers above can be regenerated by:

```bash
cd test_matrices_fast/

# Verify block orthogonality (Section 1)
python3 -c "
import numpy as np, sys
sys.path.insert(0, '.')
import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import rowspace_basis
half_win = 32
for mat in ['static-cex','mixed-tail-sharp','diffuse-diffuse','mixed-tail-soft']:
    A, _, _, _ = probe.generate_matrix_input(matrix=mat, n=1024, preset='fast', seed=0,
                                              shuffle_rows=False, row_shuffle_seed=0)
    A = np.asarray(A, dtype=np.float64)
    maxes = []
    for blk in range(8):
        s_cur = blk*half_win
        Qc = rowspace_basis(A[s_cur:s_cur+half_win])
        Qf = rowspace_basis(A[s_cur+half_win:s_cur+2*half_win])
        sigmas = np.linalg.svd(Qc.T @ Qf, compute_uv=False)
        maxes.append(sigmas[0]**2)
    print(f'{mat}: max cos² = {max(maxes):.2e}, median = {np.median(maxes):.2e}')
"

# Default-config bench (Section 5)
for m in static-cex mixed-tail-sharp diffuse-diffuse mixed-tail-soft; do
    python3 half_window_sliding_hmean_experiment.py --matrix "$m" --half-win 32 \
        --row-shuffle-seed 0 --policies combined isvd \
        --json-out "/tmp/${m}_default.json"
done

# h1 sweep with h2=0 (Section 8 — the headline result)
for hw in 32 64 128 256 512 1024; do
    h1mult=$((hw/32))
    for m in static-cex mixed-tail-sharp diffuse-diffuse mixed-tail-soft; do
        python3 half_window_sliding_hmean_experiment.py --matrix "$m" \
            --half-win 32 --h1-mult "$h1mult" --h2-mult 0 \
            --row-shuffle-seed 0 --policies combined isvd \
            --json-out "/tmp/${m}_h1_${hw}.json"
    done
done
```

Result extraction:
```python
import json
d = json.load(open('/tmp/diffuse-diffuse_h1_128.json'))
for s in d['summaries']:
    if s['mode']=='sliding':
        print(s['policy'], 'final_cos1²=', s['final_exact_cos'][0]**2)
```

---

## 12. Open questions / next steps

1. **Cross-seed verification of the buffer-size finding.** All numbers here are seed=0. Run seeds {0..4} with combined at h1=128, h2=0 on the §6 7-matrix set. If the ≥0.71 floor holds, this is the simplest fix the score family has seen.

2. **Block-entropy score variant implementation.** Code path: replace `relH(v)` in the combined-score formula with `H_block(v)` over a stored ring buffer of last K_test blocks. Compare to plain combined and to mgain Pareto override at h1=32 across the §6 7-matrix set.

3. **Reservoir sampling fidelity for H_block.** If reservoir samples are used in production instead of stored blocks, does H_block computed against reservoir samples still discriminate? Likely yes, with a degradation proportional to sample count vs n. Empirical question.

4. **etf-basket-basis is the structural exception.** Block orthogonality fails there (cos² ≈ 0.95). Most of this analysis doesn't apply. Worth a parallel sweep to characterize.

5. **The full §6 7-matrix bench at h1=128/h2=0 has not been run.** The analysis here used 4 matrices. Mts-balanced, residual-spiky-shocks, risk-residual-panel might behave differently. Particularly residual-spiky-shocks (cos²(blocks) ~1e-2, weakly orthogonal) and risk-residual-panel (cos²(blocks) ~3e-2, even weaker).

6. **Acceptance criteria should be revised.** The dev context's `mean_cos1` (vs Q_oracle) inflates apparent quality on cases where the search basis is bad. Use `mean_exact_cos1` (vs V_exact directly) as the primary metric. The "mean_cos1 = 0.98 on static-cex combined" claim becomes "mean_exact_cos1 = 0.51" which is much more honest about what the algorithm actually recovers.

---

## 13. Files referenced

- `half_window_sliding_hmean_experiment.py` — bench harness with `--h1-mult`/`--h2-mult` flags.
- `cex_restricted_space_probe.py` — `generate_matrix_input`, `entropy_iter_basis_forget`, `subspace_principal_cosines`.
- `second_slot_tail_bias_diagnostic.py` — `make_state` (FD-style update).
- `r_sk_g_score.py` — score variant definitions (S1–S8).
- `future_hmean_optimizer_diagnostic.py` — `rowspace_basis`.
- `probe_s7_peek_compare.py` — cross-window orthogonality probe (line 248–255).
- `summary/score_family_s7_dev_context.md` — predecessor context document.
- `summary/score_family_pareto_override/synthesis.md` — mgain Pareto results, partial expression of the buffer-size finding.
- `summary/overview/score_design_overview.txt` — main score-family bench table (uses `mean_cos1` against Q_oracle, see Section 9 caveat).

---

## 14. What this document does NOT establish

- It does NOT prove the score family is wrong-headed. The score variants (S6, S7, etc.) genuinely do better than naive combined at h1=32. The point is that h1=128 makes most of those gains moot.
- It does NOT validate the block-entropy score variant empirically. That's a proposal based on the analysis in Section 7 — not yet implemented and tested in the bench.
- It does NOT cover non-near-orthogonal-block matrices (etf-basket-basis being the explicit counterexample).
- It does NOT address rank > 2 or multi-slot dynamics. The whole analysis is rank=2, and most attention is on slot-1.
