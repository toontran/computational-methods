# Direction-aware carry deflation (`defsvd-recal-carry`) — iSVD-family

A new streaming right-singular-subspace policy that drops out of the
`relaxation3` score-decomposition work (see
`~/pj/meeting_focused/relaxation3_continuation_draft.md`,
`relaxation3_claims_dependencies.md`). iSVD-family only (closed form, no
optimizer). It is benchmarked here against `isvd`, `fd`, `defsvd-carryonly`,
`defsvd-symm` on the same harness (`benchmark_defsvd.py`).

## Where it comes from (the C4/C6 signal)

For iSVD the score of a direction `v` under matrix `M` is `‖Mv‖²`. Stacking
the carry `B = S_r V_rᵀ` on the new window gives `M_gain = [B; A_block]`, and by
the exact Gram identity (claim **C4**, confirmed to machine precision):

```
‖M_gain v‖² = ‖B v‖² + ‖A_block v‖²    for every v.
```

So for each carried direction `vᵢ` (column of `V_r`), the **new window's
reinforcement** of that direction is exactly

```
Δᵢ := ‖A_block vᵢ‖²   (≥ 0).
```

Claim **C6** (confirmed): in the near-orthogonal regime all `Δᵢ ≈ 0` and the
relative ordering of carried directions is unchanged; in aligned/generic
regimes `Δᵢ` is direction-dependent and the relative ordering shifts. That
direction-dependence is the signal a uniform deflation throws away.

## The gap in FD / defsvd-carryonly

FD and `defsvd-carryonly` deflate **every** carried singular value by the
**same** amount (cumulative `Δ²` in `σ²` units): `σᵢ² → max(σᵢ² − Δ², 0)`. They
do not look at which carried directions the incoming data is actually
reinforcing. The C4 signal `Δᵢ` says exactly that — and it is free (one
mat-vec per carried direction, already implied by the SVD of `M_gain`).

## The algorithm

Identical to `defsvd-carryonly` (carry deflated, **window stays pristine** —
"fresh data is sacred") except the carry's deflation is redistributed across
directions by reinforcement, **conserving the same total deflation mass**.

Per window `w` (state: `V_r` `d×r`, `S_r=diag(σ)`, cumulative `Δ²_sum`):

1. **Reinforcement on the carried basis** (the only place `Δᵢ` is clean — it
   lives on `V_r`, and `B` lives on `V_r`, so they match; this is computed
   *before* any SVD, so no basis rotation has happened yet):
   ```
   Δᵢ = ‖A_block · vᵢ‖²,   i = 1..r.
   ```
2. **Parameter-free weights** (deflate reinforced directions less). Normalize
   by the mean so the rule is scale-free and reduces exactly to uniform when
   all `Δᵢ` are equal:
   ```
   Δ̃ᵢ = Δᵢ / mean(Δ)        (if mean(Δ)=0, set all Δ̃ᵢ=0 → uniform)
   wᵢ ∝ 1 / (1 + Δ̃ᵢ),   normalized so Σ wᵢ = 1.
   ```
3. **Budget-matched redistribution.** Uniform `defsvd-carryonly` removes
   `Δ²_sum` from each of the `r` carried directions → total mass `r·Δ²_sum`.
   Redistribute that same total by the weights:
   ```
   deflᵢ = (r · Δ²_sum) · wᵢ          (Σ deflᵢ = r·Δ²_sum, identical to uniform)
   σ_def,ᵢ = √max(σᵢ² − deflᵢ, 0).
   ```
   (Excess on a direction whose `deflᵢ > σᵢ²` is clipped to 0 — that direction
   collapses and is effectively dropped, which is the intended behavior for a
   direction the data has stopped reinforcing. Clipped excess is not
   re-redistributed in this first pass; this only makes the method *more*
   conservative than uniform, never less.)
4. **Direction policy = honest iSVD on the recalibrated stack** (same shape as
   defsvd): form `B_def = (V_r · σ_def)ᵀ`, then
   ```
   M_def = [B_def; A_block]   (window pristine)
   V̂_raw = top-r right singular vectors of M_def.
   ```
5. **Honest magnitudes** (the C2 honesty rule, unchanged): recover `σ` by
   `projected_subspace_svd(M_gain, V̂_raw)` against the **raw** `M_gain`. Update
   `Δ²_sum += σ_{r+1}²(M_def)` exactly as `defsvd`.

### Properties (to verify, not assume)
- **Graceful degradation.** All `Δᵢ` equal ⇒ `wᵢ = 1/r` ⇒ `deflᵢ = Δ²_sum` ⇒
  *exactly* `defsvd-carryonly`. Near-orthogonal regime (all `Δᵢ ≈ 0`) ⇒ same.
- **No new hyperparameter.**
- **Window untouched** — does not deflate fresh data (matches the prior
  "fresh data is sacred" finding); the C4 signal is about how the window
  reinforces *old* directions, not about the window itself.

## Benchmark plan (honest stress-test ordering)

Run the **failure-prone cases first** (prior cautionary result: FD/carryonly
beat iSVD only on pure-hadamard static-cex; +5% noise flips the argmax):

1. `static-cex-noisy` (5% Gaussian noise), `static-cex-gauss`,
   `static-cex-exptail` (hardest) — **the discriminators**.
2. then pure `static-cex` (hadamard) and a few financial/structured matrices.

Report alignment `‖V_r V_rᵀ V_exact[:,:l]‖_F/√l` at **both `l=1`** (top vector)
**and `l=r`** (full subspace — where direction-aware carry deflation should
show its value), plus σ₁ rel-error and runtime.

**Discriminator kept in front:** does `defsvd-recal-carry` beat `isvd` on
gauss / noisy / exptail at l=1 AND l=r? If not on ≥2 of those 3 at l=r, say so
plainly — it then has the same hadamard-specific disease as defsvd-carryonly
and does not earn its place.
