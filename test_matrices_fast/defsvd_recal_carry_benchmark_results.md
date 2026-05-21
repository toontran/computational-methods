# `defsvd-recal-carry` — implementation & benchmark results

Spec: `isvd_recal_algorithm.md` (authoritative). Implemented as a new branch in
`benchmark_defsvd.py`; benchmarked against `isvd-ref`, `fd-ref`,
`defsvd-carryonly`, `defsvd-symm`.

Date: 2026-05-21. n=128, win=32, rank=2, seed=0, preset=fast.

---

## 1. Implementation

### Diff summary (`benchmark_defsvd.py`)

Two minimal, non-destructive edits:

1. New branch `elif mode == "DefSVD-RecalCarry":` inserted after the `DefSVD`
   branch (before `DefSVD-OrthDef`). Modeled on the `DefSVD` carry-only path,
   with the single change being **how the carry is deflated**:

   - First block (`V_r is None`): `B_def=None` — behaves exactly like iSVD,
     identical to DefSVD's first-block path.
   - Otherwise, per the spec:
     ```python
     delta_i      = np.sum((A_block @ V_r) ** 2, axis=0)     # C4 reinforcement, length r
     mean_delta   = float(np.mean(delta_i))
     delta_tilde  = 0 if mean_delta == 0.0 else delta_i / mean_delta
     w            = 1.0 / (1.0 + delta_tilde);  w /= w.sum()  # sum w = 1
     defl         = (r * delta_sum_sq) * w                    # sum defl = r * delta_sum_sq
     s_sketch_def = sqrt(max(s_sketch**2 - defl, 0))
     B_def        = (V_r * s_sketch_def).T
     ```
   - **Window stays pristine**: `A_w_def = A_block` (the `deflate_window` flag
     is ignored for this mode).
   - `M_def = [B_def; A_block]`, top-r right singular vectors `V_hat_raw`,
     `delta_this_sq = s_def[rr]**2`, honest magnitudes via
     `projected_subspace_svd(M_gain, V_hat_raw)` against the **raw** `M_gain`,
     then `delta_sum_sq += delta_this_sq`. Same accounting as DefSVD.

2. Method tuple `("defsvd-recal-carry", "DefSVD-RecalCarry", False)` appended to
   the `methods` list (after `defsvd-carryonly`). The original full list is
   preserved; nothing was removed.

### Runner / commands

A standalone runner (`run_recal_benchmark.py`) was added rather than mutating
the CLI, so `l=1` and `l=r` can both be driven through the existing
`run_streaming(..., l=...)` kwarg:

```
cd /home/ttran02/pj/computational-methods/test_matrices_fast
python run_recal_benchmark.py
```

It (a) runs the graceful-degradation assertion and (b) loops every
(matrix, method) over `l ∈ {1, rank}`. The harness CLI still works unchanged:

```
python benchmark_defsvd.py --matrices static-cex-exptail static-cex-gauss --skip-combined
```

---

## 2. Graceful-degradation verification — PASS

Verified directly from the weight formula (not a flaky matrix construction):
when all `Δ_i` are equal, `Δ̃_i = 1`, `w_i = 1/r`, and
`defl_i = (r · Δ²_sum) · (1/r) = Δ²_sum` for every i — i.e. exactly the
per-direction deflation of `defsvd-carryonly`.

Asserted for r ∈ {2,4,8} and Δ²_sum ∈ {0, 1, 7.3, 123.456}: **ALL PASS**.
`w_i == 1/r` and `defl_i == Δ²_sum` to machine precision in every case. The
`mean(Δ)==0` guard (exact `== 0.0`) yields the same uniform answer.

---

## 3. Results (n=128, win=32, rank=2, seed=0)

Alignment = `‖V_r V_rᵀ V_exact[:,:l]‖_F / √l`. Higher is better.
Discriminators listed first.

| matrix | method | align l=1 | align l=r | σ₁ rel-err | elapsed (s) |
|---|---|---|---|---|---|
| **static-cex-noisy** | isvd-ref | **0.226600** | **0.173297** | 1.74e-03 | 0.003 |
| | fd-ref | 0.174672 | 0.164629 | 7.62e-01 | 0.003 |
| | defsvd-symm | 0.190428 | 0.150012 | 1.72e-03 | 0.005 |
| | defsvd-carryonly | 0.174670 | 0.164627 | 1.75e-03 | 0.002 |
| | **defsvd-recal-carry** | 0.174670 | 0.164627 | 1.75e-03 | 0.002 |
| **static-cex-gauss** | isvd-ref | **0.181332** | **0.234597** | 1.77e-03 | 0.002 |
| | fd-ref | 0.158808 | 0.120265 | 7.60e-01 | 0.002 |
| | defsvd-symm | 0.175331 | 0.209786 | 1.74e-03 | 0.003 |
| | defsvd-carryonly | 0.158807 | 0.120264 | 1.78e-03 | 0.002 |
| | **defsvd-recal-carry** | 0.158807 | 0.120264 | 1.78e-03 | 0.002 |
| **static-cex-exptail** | isvd-ref | 0.061803 | 0.092170 | 1.20e-03 | 0.002 |
| | fd-ref | 0.123290 | 0.129287 | 8.92e-01 | 0.002 |
| | defsvd-symm | 0.074927 | 0.100463 | 1.22e-03 | 0.003 |
| | defsvd-carryonly | 0.123290 | 0.129287 | 6.23e-03 | 0.002 |
| | **defsvd-recal-carry** | 0.123290 | 0.129287 | 6.23e-03 | 0.002 |
| static-cex (hadamard) | isvd-ref | 0.149301 | 0.148235 | 1.42e-03 | 0.002 |
| | fd-ref | 0.265409 | 0.253963 | 8.74e-01 | 0.002 |
| | defsvd-symm | 0.124377 | 0.131203 | 1.42e-03 | 0.003 |
| | defsvd-carryonly | 0.265410 | 0.253962 | 3.31e-02 | 0.002 |
| | **defsvd-recal-carry** | 0.265410 | 0.253962 | 3.31e-02 | 0.002 |
| mixed-tail-sharp | isvd-ref | 0.074122 | 0.184273 | 1.22e-03 | 0.002 |
| | fd-ref | 0.468311 | 0.490992 | 8.66e-01 | 0.002 |
| | defsvd-symm | 0.040900 | 0.148597 | 9.80e-03 | 0.003 |
| | defsvd-carryonly | 0.467634 | 0.490424 | 4.12e-02 | 0.002 |
| | **defsvd-recal-carry** | 0.467634 | 0.490424 | 4.12e-02 | 0.002 |
| crowded-strategy | isvd-ref | 0.999791 | 0.998843 | 9.08e-04 | 0.002 |
| | fd-ref | 0.999819 | 0.994485 | 2.67e-01 | 0.002 |
| | defsvd-symm | 0.995520 | 0.983880 | 3.13e-03 | 0.003 |
| | defsvd-carryonly | 0.999647 | 0.976047 | 3.05e-03 | 0.002 |
| | **defsvd-recal-carry** | 0.999895 | 0.976370 | 2.16e-03 | 0.002 |
| options-vol-surface | isvd-ref | 0.999292 | **0.987394** | 3.94e-03 | 0.002 |
| | fd-ref | 0.999994 | 0.999560 | 1.30e-01 | 0.002 |
| | defsvd-symm | 0.998957 | 0.988875 | 4.58e-03 | 0.003 |
| | defsvd-carryonly | 0.999978 | 0.985326 | 2.73e-03 | 0.002 |
| | **defsvd-recal-carry** | 0.999789 | **0.862509** | 3.18e-03 | 0.002 |

Higher-rank confirmation (l=r), to show the saturation result is not a
rank-2 artifact:

| matrix | r | isvd | carryonly | recal-carry | recal − carryonly |
|---|---|---|---|---|---|
| static-cex-exptail | 2 | 0.0922 | 0.1293 | 0.1293 | +0.0e+00 |
| static-cex-exptail | 4 | 0.6963 | 0.1420 | 0.1420 | −8e-17 |
| static-cex-exptail | 8 | 0.8612 | 0.1644 | 0.1644 | −6e-17 |
| static-cex-gauss | 2 | 0.2346 | 0.1203 | 0.1203 | +0.0e+00 |
| static-cex-gauss | 4 | 0.8164 | 0.5141 | 0.5141 | +0.0e+00 |
| static-cex-gauss | 8 | 0.9865 | 0.5460 | 0.5460 | −1e-16 |

---

## 4. Verdict on the discriminator — FAILS

**`defsvd-recal-carry` does NOT beat iSVD on the discriminators, at l=1 or l=r.**

- **static-cex-noisy:** iSVD wins at both l=1 (0.2266 vs 0.1747) and l=r
  (0.1733 vs 0.1646). recal-carry loses.
- **static-cex-gauss:** iSVD wins at both l=1 (0.1813 vs 0.1588) and l=r
  (0.2346 vs 0.1203 — recal is ~half iSVD's subspace alignment). recal-carry
  loses badly, worsening as rank grows (r=8: iSVD 0.986 vs recal 0.546).
- **static-cex-exptail:** recal-carry's l=1 (0.1233) and l=r (0.1293) edge out
  iSVD at r=2, but this reverses sharply with rank: at r=4 iSVD is 0.696 vs
  recal 0.142, at r=8 iSVD is 0.861 vs recal 0.164. No honest claim of a win
  on exptail survives past r=2.

So it loses on ≥2 of the 3 discriminators (in fact all 3 once rank is raised).
**Same hadamard-specific disease as `defsvd-carryonly`** — it only "wins" on the
pure-hadamard `static-cex` (0.265 vs iSVD 0.149) and structured financial
matrices, and loses the moment the signal basis is noisy/gaussian/exp-tailed.

### Why it is identical to `defsvd-carryonly` on the discriminators

This is the central finding. On the discriminators recal-carry is **numerically
identical to carryonly to machine precision** — and not because the C4 weights
are uniform. Measured weights are non-uniform (e.g. exptail r=2: w=[0.468,
0.532]; r=4: w varies 0.32–0.40 across directions). The reason is **deflation-
budget saturation**:

On the restricted-space matrices the carried `σ_i²` are all ≈ 0.97 (a flat
tail), while the cumulative redistribution mass `r · Δ²_sum` grows to 3.9, 7.7,
11.6 by windows 2/3/4 (r=4 exptail). Every per-direction `defl_i` therefore
exceeds every `σ_i²`, so `max(σ² − defl, 0)` clips the **entire carry to zero
regardless of how the mass is distributed**. The redistribution has nothing left
to redistribute. The spec's "graceful degradation" pathway is the *only* pathway
that fires on these matrices — via saturation, not via uniform Δ. recal-carry
inherits carryonly's real problem, which is that its deflation budget is the
wrong scale for these matrices to begin with.

### A separate, active harm: options-vol-surface

On `options-vol-surface` the carry is **not** saturated (`defl < σ²` at every
window: e.g. σ²=[21.3, 7.5], defl=[1.3, 2.5]). Here the weighted redistribution
genuinely differs from uniform — and it is **worse**: l=r drops from 0.985
(carryonly) / 0.987 (iSVD) to **0.863**, a >12-point regression. The weights
load more deflation onto the smaller-σ carried direction (w≈0.66 onto direction
2), over-collapsing the weaker direction and corrupting the l=r subspace. So in
the one regime where the recal scheme actually does something, it does something
harmful. (l=1 is unaffected: 0.9998.)

### Bottom line

The C4 reinforcement signal is real and the weights are non-degenerate, but the
method does not earn its place: where the budget saturates (the discriminators)
it collapses to `defsvd-carryonly` and shares its hadamard-specific disease;
where the budget does not saturate (options-vol-surface) the redistribution is
actively harmful at l=r. No new hyperparameter — but also no new benefit on the
discriminators, and a downside on a financial matrix.
