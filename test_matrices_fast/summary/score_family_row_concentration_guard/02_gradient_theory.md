# FAM-02: Gradient theory

Date: 2026-04-28
Companion: `01_family_design.md`, `r_sk_g_score.py`.

This is the shared gradient infra all D-variants plug into. We do NOT introduce
Stiefel ascent here (that is FAM-01); the family stays scalar-per-v on the
sphere.

## Notation

- `A_sk = state.s · state.V^T` — rank-r carry matrix (n_carry × n).
- `A_c = A_cur` (h × n), `A_f = A_fut` (h × n).
- `y_X = A_X v` for X ∈ {sk, c, f}; `raw_X = ‖y_X‖²`.
- `W_sk = sk_F2_low = ‖A_sk‖_F²` (rank-r CARRY, NOT full prefix; overview §2(c)).
  `W_c = cur_F2`, `W_f = fut_F2`.
- `u_sk = raw_sk / W_sk`, `u_g1 = raw_c / W_c`, `u_g2 = raw_f / W_f`.
- `S6(v) = HM3(u_sk, u_g1, u_g2)` (sketch present) or `HM2(u_g1, u_g2)`
  (block-1 fall-through), with `HMk = k / Σ_X 1/u_X`.
- `relH1(v) = H(p) / log(h)` where `p_i = (A_c v)_i² / Σ_j (A_c v)_j²` and
  `H(p) = -Σ p_i log p_i`. Implemented in
  `hmean_evidence_score.entropy_relH1_value_grad`.

## D0: `S6(v) · relH1(v)`

### Score

```
score_D0(v) = S6(v) · relH1(v)
```

Both factors are smooth on the sphere wherever (a) all u_X > 0 (so HM3 is
finite) and (b) `A_c v` is not zero (so `p` is well-defined). On the
streaming domain these conditions hold for any v whose response in A_cur is
nonzero — which is the same domain S6 is well-defined on.

### Gradient (product rule)

```
∇ score_D0 = relH1(v) · ∇S6(v) + S6(v) · ∇relH1(v)
```

Both `∇S6` and `∇relH1` are already implemented and verified
(`r_sk_g_score.r_sk_g_value_grad` for S6, `hmean_evidence_score
.entropy_relH1_value_grad` for relH1). The product-rule gradient adds one
multiplication per factor; no new derivative code is required.

### Implementation pattern

```python
# In r_sk_g_value_grad branch for variant == "D0":
relH1, grad_relH1 = entropy_relH1_value_grad(A_c, v)
# Compute S6 score and gradient as in the existing S6 branch:
S6, grad_S6 = ... (HM3 / HM2 logic)
score = S6 * relH1
grad  = relH1 * grad_S6 + S6 * grad_relH1
```

This matches the pattern S4 / S5 already use for `HM3 · relH1` (see lines
221–222 and 262–263 of `r_sk_g_score.py`). The only difference for D0 is
that the inside is the F-weighted S6 form, not the raw-norm S4/S5 forms.

### Block-1 fall-through

S6 falls through to `HM2(u_g1, u_g2)` when no sketch is present. D0 inherits
the same fall-through:

```
score_D0_b1(v) = HM2(u_g1, u_g2) · relH1(v)
```

The product rule still applies; grad is `relH1 · grad_HM2 + HM2 · grad_relH1`.

### FD check

- Region: same as S6 (all u_X > 0 on a random unit v with the standard test
  matrices). No special carry-aligned probe needed (relH1 > 0 generically).
- Acceptance: rel_err < 1e-7 at float64 (same bar as S6).
- Probe layout reuses `r_sk_g_score.gradient_check`: 20 random coordinates,
  central FD with h = 1e-6. The S6 branch passes at ~1e-10; D0 should land in
  the same regime since both factors are well-conditioned.

## D1: leverage-weighted u_X (SPEC ONLY — later agent)

For X ∈ {c, f}, define a per-row leverage weight `ℓ_X[i]` (e.g., row-norm of
A_X projected onto state.V, or the i'th row of `A_X · state.V`'s norm). Then
replace

```
u_X(v) = Σ_i (A_X[i,:] v)² / W_X
       → u_X^lev(v) = Σ_i ℓ_X[i]^η · (A_X[i,:] v)² / Σ_i ℓ_X[i]^η
```

with η ∈ {0, 1/2, 1} as a hyperparameter. η=0 reduces to S6.

The gradient picks up an extra factor `ℓ_X[i]^η` inside the chain
`coef · (1/u_X^2) · (1/W_X^lev) · (2 · A_X^T diag(ℓ_X^η) y_X)` — closed-form,
but requires the η weight to be a per-block constant (fix it at block start).

Risk: choice of `ℓ_X` is not well-pinned by current evidence; INFRA-06
(carry-trajectory probe) needs to land first.

## D2: soft sigmoid gate on u_sk (SPEC ONLY — later agent)

```
score_D2(v) = S6(v) · σ(κ · (u_sk(v) - τ))
```

with `σ(x) = 1/(1+e^{-x})`, κ a sharpness hyperparameter, τ a per-block
threshold (e.g. τ = 0.5 · max_v u_sk(v) over the search basis or τ = 1/r as
the "fair share" of average direction energy).

Gradient: product rule plus chain rule for σ. The σ factor's grad is
`κ · σ · (1-σ) · ∇u_sk`, and `∇u_sk = (2/W_sk) · A_sk^T y_sk`. Smooth
everywhere; degenerate only at the limits (σ → 0 or 1).

Hyperparameter sweep over (κ, τ) — would benefit from INFRA-08 (sweep
harness). Cheaper at fixed (κ, τ) = (5, 1/r).
