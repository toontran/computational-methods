# FAM-02: Row-concentration guard — family design

Date: 2026-04-28
Status: D0 in-progress this round; D1, D2 spec-only stubs for later agents.

## Hypothesis

The spiky-residual regime (residual-spiky-shocks, risk-residual-panel) fails
under S6 / online because the score allows v's whose `A_cur v` energy
distribution is dominated by a single row (low relH1). Such v's score high on
the per-window "fraction of available energy" of S6 but generalize poorly to
A_fut and to subsequent blocks. iSVD wins because its update is integrative
across blocks rather than per-block-energy-maximizing.

A row-concentration guard re-introduces the relH1 entropy factor that S6
deliberately dropped (overview §2(d)), but in a form that keeps the score
smooth on the sphere wherever both factors are positive (no S5-style hard
gate / oracle exclusion).

## Variant table

| variant | score | hyperparams | targets | predicted effect | scope |
|---|---|---|---|---|---|
| **D0** | `S6(v) · relH1(A_cur v)` | none | Q3 (relH1 multiplicative) | mixed-tail-sharp / residual-spiky-shocks: tighter basin around higher-entropy candidates; expected lift on residual-spiky cos1², small or zero loss on tail-dominant where the optimum is already balanced | THIS ROUND |
| D1 | `HM3(u_sk, u_g1_lev, u_g2_lev)` where `u_g_lev` weights row energies by row-leverage scores from `A_sketch` (so a single dominant-leverage row counts more / less than a single noise row, depending on choice) | leverage exponent η ∈ {0, 1/2, 1} | Q4 (per-row leverage / c-weighting) AND Q3 simultaneously | replaces the multiplicative entropy with a structural per-row weighting; predicted to be more discriminating on risk-residual-panel where the spike *is* high-leverage. Alternative to D0 if D0 fails on tail-dominant matrices. | LATER AGENT |
| D2 | `S6(v) · sigmoid(κ · (u_sk(v) − τ))` with τ=0.5·E[u_sk over reachable v's], κ ∈ {2, 5, 10} | κ, τ | F2 in the f_hm3 spec; smooth alternative to S5's hard gate | locks v away from low-u_sk basins without entropy assumption; useful when the failure mode is "low carry alignment" rather than "low row entropy" | LATER AGENT |

## What each variant tests

- **D0** tests whether relH1 *as a multiplicative correction* is enough to
  recover the spiky regime, with no other change to S6. It's the cheapest
  possible row-concentration guard. If D0 ships, Q3 closes; if D0 fails on
  tail-dominant matrices but wins on spiky-residual, that motivates a hybrid
  policy (D1 leverage variant, or matrix-class-conditional dispatch).
- **D1** tests whether the row-concentration signal should live INSIDE u_X
  (changing what "captured energy" means) rather than as an outside multiplier.
  Closer to the structural fix Q4 suggests.
- **D2** tests whether the failure mode is row-concentration (D0/D1) or
  carry-alignment (D2). D2 is closer in spirit to S5 but smoother.

## Predictions

D0 prediction (this agent's job to verify or refute):

- residual-spiky-shocks: cos1² climbs from S6's 0.266 toward iSVD's 0.637.
  relH1 disqualifies low-entropy v's that S6 currently picks.
- risk-residual-panel: similar lift. Online's 0.064 cos1² there is the
  multi-row entropy failure mode mentioned in toolkit §7.
- mixed-tail-sharp / mixed-tail-balanced / static-cex: NEUTRAL or small loss.
  S6 already finds high-score v's in these matrices; multiplying by
  relH1 ∈ [0, 1] doesn't change the ranking unless relH1 strongly differs
  between the optimum and the oracle. It might even tighten the basin (P4).
- diffuse-diffuse: NEUTRAL or small lift. Diffuse rowspace ⇒ relH1 ≈ 1 for
  most v's; multiplier acts ~uniformly.
- etf-basket-basis: small risk of regression (S6 wins decisively here);
  watch for relH1 of S6's optimum being notably below 1.

## Family-shared infra

D1 and D2 reuse the same gradient framework:

- All three variants are `f(v) · g(v)` products (or weighted means) of S6
  arguments — gradient is one product-rule term per factor.
- All three pass T1 grad-check via `r_sk_g_score.py --gradient-check`.
- All three are wired into `half_window_sliding_hmean_experiment.py` as
  `--rsk-variant` choices (D0 → "D0", etc.).
- Per-block T2 reuses the existing 3-matrix probe set from the toolkit.

## Sequencing

D0 first (this round). If D0 ships, run D2 (cheap). If D0 fails on tail-
dominant matrices, run D1 (structural alternative).

## Scope discipline

D1 and D2 are explicitly **spec-only** in this round. Do NOT implement them
without first reading D0's `synthesis.md` — the verdict there changes which
variant to prioritize and may change the score forms above.
