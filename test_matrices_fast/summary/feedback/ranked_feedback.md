# Ranked Feedback Triage

Source feedback: comprehension, scientific critique, evidence rigor, methodology, logic, decision relevance, and math correctness advisor outputs.

## Fatal

1. **Fatal (Certain, Method/Core): `B_union := range((B; A_w; A_fut))` is dimensionally wrong as written.**
   The stacked matrix has column space in row-coordinate space, not a subspace of `R^d`. The intended audit domain must be `rowspan((B; A_w; A_fut))` or `range((B; A_w; A_fut)^T)`. This affects the formal definition of the frame audit and every claim phrased as optimization over `B_union`.

2. **Fatal (Certain, Method): uniform scaling of `u_sk` does not preserve HM argmaxes.**
   The report claims differing constants only rescale `u_sk` uniformly in `v` and leave argmaxes unchanged. That is false for `HM_3(u_sk,u_cur,u_fut)`: rescaling one slot changes the tradeoff against the other slots and can move the maximizer.

## Major

3. **Major (Likely, Core): M1/M2/M3 are not orthogonal mechanisms.**
   The feedback consistently reads them as symptoms of one issue: a window-local surrogate ranks directions differently from the leading invariant subspace of global `A^T A`. Treating them as independent causal axes risks overfitting separate repairs to path-dependent symptoms.

4. **Major (Certain, Core): "no further scalar weight tuning fixes this" is stronger than the evidence.**
   The report supports "E2-style balance tuning failed on selected cases," not all scalar weights, schedules, normalizations, additive forms, or coupled frame objectives.

5. **Major (Likely, Core): V1/V2/V3 do not exhaust cheap feature additions.**
   The tested future-direction and current-future correlation terms fail, but the conclusion should not imply all low-intrusion window-local features are ruled out. The current evidence covers two specific additions on three block-31 regression matrices.

6. **Major (Certain, Core/Method): V1 explanation is contradicted by the table.**
   V1 is said to lower `S_1(Z_oracle)` and raise the non-oracle argmax, but Appendix B.1 shows the oracle score rises on two of three matrices and the joint score falls on all three. The fail conclusion survives; the mechanism does not.

7. **Major (Certain, Core): E2 "regresses sharply on the same matrices" is contradicted by own capture table.**
   E2 is worse on `diffuse-diffuse` and `residual-spiky-shocks`, but better on `mixed-tail-soft` in the cited seven-matrix capture table. The claim needs either a different metric or narrower wording.

8. **Major (Likely, Method): optimizer certification is overstated.**
   Small `j-g` shows greedy and joint routines found similar values, and a non-oracle frame beating the oracle is enough to refute oracle-as-maximizer locally. It does not certify the global maximum of a nonconvex Stiefel objective.

9. **Major (Likely, Assumption/Core): the V1-V4 conclusion is underpowered and selected.**
   The main frame-sufficiency result uses three regression matrices, one block, rank 2, and unspecified restart counts. That is adequate as a regression gate, not as a general statement about rank-2 window-local feature insufficiency.

10. **Major (Likely, Method): V2's current-future cross term may not test a principled geometric quantity.**
    `<A_cur Z, A_fut Z>_F` assumes matched row counts and meaningful row pairing. Without that, V2 may be testing row-coordinate coincidence rather than shared right-subspace geometry.

11. **Major (Likely, Core): target definitions shift between slotwise vectors, oriented frames, and spans.**
    Several conclusions move between `v_i`, `Z_oracle`, `span(V_2)`, and `V_r`. V4's residual oriented-frame gap is not necessarily a failure if the actual target is span recovery.

12. **Major (Likely, Method): M2 sketch-dominance claim lacks the per-direction decomposition needed.**
    State alignment rises strongly, but the report does not show `||Bv||^2`, `||A_wv||^2`, and `phi_w(v)` at winner vs oracle. Aggregate `rho_past` and small `rho_B` do not prove pointwise sketch-energy dominance.

## Moderate

13. **Moderate (Certain, Presentation/Core): Appendix A.4's "S6 argmax is the oracle" conflicts with later frame-level HM failure language.**
    This may be vector/anchored vs frame-level terminology, but the distinction is load-bearing and currently easy to misread.

14. **Moderate (Likely, Method): row-cheat diagnostics are used as an acceptance concept before they exist.**
    H3 invokes row-cheat dominance as a screen, but the frame-level row-cheat diagnostic is still open work. Current H3 conclusions should not rely on that gate.

15. **Moderate (Likely, Method): exact `Z_oracle` argmax may be too brittle as the success metric.**
    If terminal capture or principal angles are the operational goal, score gap, oriented-frame recovery, and span recovery need to be linked empirically.

16. **Moderate (Likely, Method): future/lookahead is under-ablated.**
    The report allows `A_fut` but does not vary future-window length, offset, or independence. This leaves ambiguity between "future evidence is absent," "future evidence is insufficient," and "the scoring form mishandles future evidence."

17. **Moderate (Certain, Assumption): many formulas need explicit zero-denominator and tie conventions.**
    `h_2`, `phi_w`, HM slots, `sigma_{k(v)}`, projections, and normalizations require nonzero denominators; `k(v)` needs tie handling. These are mostly edge cases but affect formal correctness.

18. **Moderate (Likely, Method): block/time and rank controls are missing.**
    The feedback recommends repeating key frame audits across early/middle/late blocks and a small rank ladder before treating the failures as representative.

19. **Moderate (Likely, Core): `diffuse-diffuse` score discrepancy remains a confound.**
    Earlier `~0.297` vs Appendix A.4 `0.2164` does not appear to reverse the high-budget gap, but it affects a load-bearing example and should be reconciled for reproducibility.

20. **Moderate (Likely, Method): M3 future-row success is confounded by oracle-warm initialization.**
    The `0.999` result shows future rows can help in a controlled block-1 setting, but does not by itself establish deployable recovery or rule out initialization/restart effects.

## Minor

21. **Minor (Certain, Presentation): row-count substitution `row(N_w) -> k=row(C_w)` needs a named approximation.**
    Add a sentence explaining that `k` is an empirical pooled-row proxy used only in score experiments.

22. **Minor (Certain, Presentation): define `B_union` before first use in `oracle_proj`.**
    The notation is currently introduced after it is needed.

23. **Minor (Certain, Presentation): make indexing conventions explicit early.**
    The report alternates `v_1^*`, `v_2^*`, `cos_1^2`, and zero-based appendix labels. State the convention near the first oracle definition.

24. **Minor (Certain, Presentation): define shorthand at first use.**
    `AB-03`, `DIAG-05`, `force_O2`, `force_frame`, and "regression matrices" need local parentheticals.

25. **Minor (Certain, Presentation): replace/remove the explicit placeholder in "common thread."**
    This is not conceptually important, but it interrupts a polished report.

## Recommended Fix Order

1. Repair the two mathematical correctness errors: `B_union` and HM argmax scaling.
2. Narrow overclaims: scalar tuning, cheap additions ruled out, V1/V2 mechanisms, E2 regression, and global insufficiency.
3. Separate targets explicitly: value lower bound, slotwise direction, oriented frame, and unordered subspace.
4. Add the missing evidence tables/controls for M2, optimizer restarts, block/rank variation, and row-cheat.
5. Promote the smallest `A^T A`/Gram proxy frame audit as the next decision-changing experiment.
