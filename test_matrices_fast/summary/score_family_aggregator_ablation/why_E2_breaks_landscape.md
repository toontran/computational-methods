# Why S6_E2 breaks the score landscape

Date: 2026-04-28
Backlog: AB-03 phase 1 closure; followup to `synthesis_S6_E2.md`
Investigators: probe_e2_landscape.py, force-oracle-v{2,frame} bench

## Verdict

The DIAG-04b oracle u-balance prediction failed because **balance under
E2 weights does not coincide with the score's argmax under E2 weights.**
The S6_E2 score's argmax in B_union has been moved by the per-direction
reweighting to a non-oracle point. The optimizer correctly finds that
non-oracle argmax — and that argmax has its own (different) balanced
u-vector that DIAG-04b's oracle-evaluated metric never measured.

A second, separable issue is exposed by the force-oracle bench: the
streaming carry V_default[:,0] is itself a poor approximation of
oracle_v1 on the diffuse / residual matrices, so even forcing the slot-2
pick to oracle_v2_proj cannot recover cos1² — the SVD-frame
[V_default[:,0], oracle_v2_proj] simply does not contain the oracle plane.

Hypothesis triage (from synthesis_S6_E2.md):

- **H1 — landscape-argmax-off-oracle: confirmed (load-bearing).** At b31,
  Δ_S6_E2(slot-2 winner − oracle_v2_proj) = +0.041 on diffuse-diffuse,
  +0.047 on residual-spiky-shocks, while Δ_S6 = +0.007 on both. The
  optimizer reaches a strictly higher S6_E2 score than the oracle does;
  under S6 the oracle is essentially the argmax.
- **H2 — Voronoi step changes from per-direction argmax: weak.** On
  the geodesic between oracle_v1_proj and oracle_v2_proj we logged
  argmax(k_sk, k_g1, k_g2) at every t. Score jumps at boundary
  transitions: max-Δ ≈ 1.8e-3 (residual) to 7.2e-3 (mixed-tail-soft)
  under E2 — non-zero but two orders of magnitude below the score
  variation along the geodesic. Not the dominant mechanism.
- **H3 — slot-1 collapse from removing F-norm sigma prior: not
  supported.** Under E2 the score gradient near sketch_v1 is *stronger*
  (|grad_E2| ≈ 30–100× |grad_S6|) and the back-pull toward sketch_v1
  is positive and larger under E2 than S6. The reweighting amplifies
  the score everywhere, but the relative pull toward sketch_v1 is not
  weakened. The cos²(S6_E2_v1*, sketch_v1) is actually *higher* than
  cos²(S6_v1*, sketch_v1) on diffuse-diffuse (0.342 vs 0.121) and
  mixed-tail-soft (0.347 vs 0.116).

## Probe A — oracle-vs-winner score gap (b31)

Read `landscape_<matrix>_b31.txt` for full per-candidate tables.
Highlights:

| matrix | score_S6_E2(o2) | score_S6_E2(v2*) | gap_E2 | gap_S6 |
| --- | ---: | ---: | ---: | ---: |
| diffuse-diffuse        | 0.297 | 0.338 | **+0.041** | +0.007 |
| residual-spiky-shocks  | 0.307 | 0.354 | **+0.047** | +0.007 |
| mixed-tail-soft        | 0.333 | 0.338 | +0.006 | +0.004 |

On the two regression matrices, S6_E2's argmax in B_union is 14–16%
above the oracle in score. On mixed-tail-soft (a §6 *gainer*) the gap
is small — and the geodesic argmax(S6_E2) sits at t/π = 0.500 with
cos²(o1, o2) = (0.000, 0.997), i.e., essentially on oracle_v2_proj.
That is why mixed-tail-soft GAINS under E2 (Δcos1² = +0.176): on this
matrix the new landscape's argmax happens to coincide with the oracle.

## Probe B/C — geodesic landscape

The S6 score on the oracle plane is monotone or single-peaked very
close to the oracle endpoints (argmax(S6) at t/π = 0.43–0.50). Under
S6_E2 the peak shifts inward (e.g., diffuse-diffuse: 0.429 → 0.475;
residual: 0.329 → 0.375), and the optimizer's actual winner is OFF
the oracle plane entirely (cos²(o1) and cos²(o2) both small at the
S6_E2 v2* winner). The reweighting reshapes the landscape so the
global argmax leaves both the oracle and even the 2D oracle plane.

## Probe D — slot-1 attractor

Not the regression mechanism. Score gradient magnitude near
sketch_v1 grows under E2 (because per-direction weights are smaller
than the F-norm sum, so u_X is larger), and the gradient still pulls
toward sketch_v1 with a comparable or larger positive sign. There is
no observable "slot-1 collapse" in the gradient field at b31.

## Probe E — force-oracle bench (closure)

To address the user's question — *"is the problem really about
balance? we can use oracle to optimize for the right balance"* — we
added two bench overrides:

- **--force-oracle-v2**: chosen_v2 ← oracle_v2_proj (V_default[:,0]
  from streaming).
- **--force-oracle-frame**: V_selected ← rank2_svd_frame(oracle_v1_proj,
  oracle_v2_proj, M_gain).

Sliding b31 oracle-mass capture (cos0² + cos1², max = 2):

| matrix | S6 | E2 | force_O2 | force_frame (ceiling) |
| --- | ---: | ---: | ---: | ---: |
| diffuse-diffuse        | 1.05 | **0.66** | 1.00 | 1.95 |
| residual-spiky-shocks  | 1.18 | 1.02 | 1.01 | 1.92 |
| mixed-tail-soft        | 1.08 | **1.55** | 1.00 | 1.95 |
| mixed-tail-sharp       | 1.07 | 1.09 | 1.00 | 1.95 |
| static-cex             | 1.12 | 1.05 | 1.00 | 1.95 |
| mixed-tail-balanced    | 1.09 | 1.19 | 1.00 | 1.95 |
| etf-basket-basis       | 1.64 | 1.54 | 2.00 | 2.00 |

What this tells us:

1. **The cos² ceiling at b31 is ≈1.95 on every §6 matrix.** Both
   oracle directions ARE in rowspan(M_gain). The pipeline's
   structural limit at half_win=32 is high — we are not near it.
2. **Forcing chosen_v2 = oracle_v2_proj gets capture ≈1.00** on six
   of seven matrices. That is exactly *one* direction captured. Why?
   Because V_default[:,0] (combined-score streaming carry) has small
   cos² with V_exact[:,0] on these matrices. The 2-plane spanned by
   [V_default[:,0], oracle_v2_proj] catches V_exact[:,1] perfectly
   and V_exact[:,0] not at all. Only on etf-basket-basis (where the
   carry happens to align well) does fO2 hit 2.00.
3. **Hence "use the oracle to balance" cannot, by itself, fix the E2
   regression.** On diffuse-diffuse and residual-spiky-shocks the E2
   optimizer's *non-oracle* winner spans a richer 2D plane (paired
   with V_default[:,0]) than the oracle does. fO2 capture (1.00) is
   *worse* than even the E2 capture (1.02 / 0.66). The balance-target
   the oracle satisfies is geometrically narrower than the optimizer's
   blend.
4. **The full oracle frame recovers the ceiling everywhere**, so the
   regression IS recoverable in principle — but only if BOTH slot-1
   anchor and slot-2 pick land on the oracle plane. Replacing one
   without the other doesn't help on most matrices.

## Principle (for score_design_overview.txt §1quater)

> An oracle u-balance metric measures a property of the oracle's
> u-vector under a candidate weight scheme. It does NOT measure where
> the score's argmax actually sits under that scheme. A new weighting
> can balance the oracle and simultaneously move the score's argmax
> off the oracle by a margin the audit metric cannot see (since the
> audit only evaluates AT the oracle). Before promoting a weighting
> from a balance audit to a ship variant, run the §6 sliding-cos²
> screen — this is the only test that exposes argmax-relocation.

## Screening recommendation (FAM-01 / FAM-04 / future weightings)

Any new score variant proposed on the basis of an audit metric (oracle
balance, exploitability, frame match, etc.) must additionally pass:

1. **Oracle-vs-winner gap test** at b31 on the §6 suite:
   `Δ_score(winner − oracle_v2_proj) ≤ Δ_score(S6_HM3, winner − oracle_v2_proj)`
   (i.e., the new score must not move the argmax further from the
   oracle than S6 does). Implemented in `probe_e2_landscape.py`,
   takes ~30 s per matrix.
2. **§6 sliding cos² bench** on at least 7 §6 matrices, with the
   ship-screen `Δcos1² ≥ −0.05` on every matrix.
3. **Carry-anchor check** (suggested for FAM-01 rank-r lift): even a
   correctly-placed slot-2 winner cannot recover cos1² when
   V_default[:,0] is bad. Any rank-2 score variant must therefore
   either (a) replace V_default[:,0] in the optimizer's basis (joint
   rank-2 lift) or (b) accept the structural ceiling cos²(V_default,
   V_exact[:,0]) imposed by combined-score streaming.

## Cross-references

- Per-matrix probe outputs: `landscape_<matrix>_b31.{txt,json}`,
  `landscape_summary_b31.txt`
- Force-oracle bench outputs:
  `forceO2_<matrix>_win64.{txt,json,csv,log}`,
  `forceFrame_<matrix>_win64.{txt,json,csv,log}`
- Closure table: `closure_table_b31.txt`
- Source: `probe_e2_landscape.py`, `probe_e2_landscape_summary.py`,
  `half_window_sliding_hmean_experiment.py` (--force-oracle-v2,
  --force-oracle-frame flags)
- Synthesis: `synthesis_S6_E2.md`
- DIAG-04b basis: `summary/infra_oracle_u_balance/scheme_comparison.md`
- Workflow: `summary/overview/score_family_workflow.txt` [AB-03]
