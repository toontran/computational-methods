# Logical-Structure / Argument-Flow Audit

Source DOC B: `summary/overview/score_design_overview.txt`

Background consulted: `../reports/approximation/new_approx_combined.txt`

Advisor lens: premises -> intermediate claims -> conclusions; hidden assumptions, non-sequiturs, circularity, inconsistent definitions, and over-strong conclusions.

## Fatal / Major Flags

No Fatal leak found. The document is internally aware of most of its current weaknesses. The Major leaks below are places where the current conclusion still depends on a stronger premise than the overview establishes.

### L1. Scalar level-set dimension does not itself prove the plateau mechanism

Severity: Major  
Confidence: High  
Scope: Sections 3, 5, 6, 7/Q1-Q2/Q12

Claim: "Any scalar-per-v score on the union sphere has a level set of codimension 1"; therefore the deep problem is a wide plateau, and rank-r/subspace scoring is the structural fix.

Dependency problem: codimension-1 level sets are generic for scalar functions and do not by themselves imply a flat high-score ridge, weak oracle identifiability, optimizer drift, or failure of scalar scores. The empirical premise is stronger and more relevant: implemented optimizers find non-oracle points with score above the projected oracle. The dimension argument alone is a non-sequitur unless paired with a quantitative flatness/curvature/near-level-set measure.

Tighter version: "For the tested S6 landscapes, high-score non-oracle regions are empirically broad enough that the optimizer consistently leaves the oracle; this motivates rank-r diagnostics. A formal scalar-score limitation remains Q12."

### L2. "Rank-r lift may shrink the plateau" is not entailed by the stated geometry

Severity: Major  
Confidence: Medium-High  
Scope: Sections 5, 7/Q1, priority order

Claim: moving to rank 3 or 4 may quietly fix mixed-tail-sharp because "the plateau dimension shrinks."

Dependency problem: the document does not derive this dimension comparison. A rank-r Grassmann/Stiefel search generally changes both the manifold and the score; its dimension can increase with r over the relevant range. The argument can still be a good experiment, but the stated geometric reason is unsupported. What rank-r could fix is not obviously "smaller plateau"; it may instead fix greedy anchoring, use frame interactions, or change the target from oriented slot to subspace.

Tighter version: "Rank-r lift is motivated because the operational target is a subspace and because greedy slotwise deflation may create anchoring artifacts; whether it reduces plateau ambiguity is an empirical/theoretical question."

### L3. "Slot-1 is solved; the gap is entirely slot-2" conflicts with later force-oracle evidence

Severity: Major  
Confidence: High  
Scope: Sections 1, 1quinquies, 6, Q0

Claim: slot-1 is essentially solved by combined-step, and the operational gap is entirely slot-2 / P4 plateau drift.

Dependency problem: later text says `--force-oracle-v2` does not recover six of seven matrices because `V_default[:,0]` is itself a poor approximation of oracle slot 1; only `--force-oracle-frame` recovers the ceiling. Section 6 also reports nontrivial S6 cos0 failures on diffuse-diffuse, mixed-tail cases, and residual-spiky-shocks. Thus "slot-1 solved" is at best local to a subset of blocks/matrices or to the initial combined-step construction, not a global premise for the streaming conclusion.

Tighter version: "Slot-1 is often less broken than slot-2 and may be well determined in some local diagnostics, but terminal recovery still depends on joint frame anchoring."

### L4. The online baseline is both a ceiling and an operational target

Severity: Major  
Confidence: High  
Scope: Sections 1, 6, Q0

Claim: S6 must close the gap to `future_hmean_online`, while the document also says the reported online table is oracle-aware and must be treated as a ceiling until the value-only rerun lands.

Dependency problem: the document correctly adds the caveat, but several downstream readings still lean on the oracle-aware table: "online is far ahead," "operational gap," and the strict Q0 comparison language. Until value-only online numbers are propagated, the comparison can motivate a ceiling analysis but cannot support a production-target conclusion.

Tighter version: "Use oracle-aware online only as an upper bound and diagnostic for what a candidate pool can achieve; defer all close-the-gap claims to the value-only rerun."

### L5. "High-sample-entropy home turf" is load-bearing but sometimes shifts between matrix-level and slot-level claims

Severity: Major  
Confidence: Medium-High  
Scope: Sections 1ter, 6 regime classification, Q10

Claim: mixed-tail-sharp and the mixed-tail family are high-sample-entropy home turf where the score family must win, while M1 also says mixed-tail-sharp oracle slot-2 is genuinely concentrated and is a boundary regime where neither family wins.

Dependency problem: "high-sample-entropy" is used at matrix, slot, oracle-response, and candidate-response levels. These are not interchangeable. A matrix can have a high-support slot 1 and a low-support slot 2, or vice versa. The document partly corrects this for residual-spiky-shocks but leaves mixed-tail-sharp carrying both labels in different arguments.

Tighter version: classify regimes per `(matrix, slot, block, target)` rather than per matrix. Then state which cells are home turf, boundary, or SVD territory.

### L6. HM3 is described as a high-sample-entropy bias, but its stated variables only enforce magnitude balance

Severity: Major  
Confidence: High  
Scope: Sections 1ter, 2, 6b, Q3/Q13/Q14

Claim: phi, HM3, and relH1 are one high-sample-entropy inductive-bias family.

Dependency problem: phi and relH1 directly encode within-window spread/support-like information. HM3, as defined, uses only normalized magnitudes `||A_X v||^2 / ||A_X||_F^2` across carry/current/future; it enforces between-source balance, not row entropy. A direction can have balanced high magnitude in current and future for unreliable or low-support reasons, which the document itself later recognizes under "magnitude-only proxies."

Tighter version: "HM3 is a split-source magnitude-stability bias; phi/relH1 are within-window spread/support biases. The proposed unification is reliability/stability, not entropy strictly."

### L7. Tail-conspiracy unification is stronger than the diagnostic premise

Severity: Major  
Confidence: Medium  
Scope: Sections 1bis, 1ter, Q11

Claim: M1/M2/M3 are sub-cases of one tail-conspiracy / reliability failure.

Dependency problem: DIAG-05 is reported to separate score-favored directions from oracles by at least one of relH1, current/future asymmetry, or Pearson replicability. That supports "visible evidence is not reliably oracle-identifying." It does not necessarily establish the narrower causal story "tail conspiracy," especially for carry-pinning M2 where the mechanism can be state retention/information loss rather than row-tail collusion.

Tighter version: "M1/M2/M3 share a broader reliability/identifiability failure; tail conspiracy is one important subtype and should be reserved for cases with row-tail or support evidence."

## Dependency Graph

### Root Definitions And Targets

**D1. Streaming target.** At each block, the method commits a rank-r right subspace and is judged by terminal `final_exact_cos`, especially slot-2 cos².

**D2. Feasible local evidence.** The score may use the carry/sketch, current half-window, and future half-window, but not `V_exact`.

**D3. Combined score.** DOC A derives `score_w(v) = (||Bv||² + ||A_w v||²) * phi_w(v)` from assumptions A1/A2 plus a lower-bound-style inequality. The overview now treats this as algebraically derived but empirically unreliable because A2 is not expected to hold on the counterexample suite.

**D4. S6 score.** `S6(v) = HM3(u_sk, u_g1, u_g2)` when a sketch is present, and `HM2(u_g1, u_g2)` at block 1. The `u_X` terms are Frobenius-normalized magnitude fractions.

**D5. Current controlling principle.** The question is not whether the oracle has good balance, but whether the visible evidence makes the oracle vector/frame identifiable as the score maximizer in the relevant search domain.

**D6. Target distinctions.** Slotwise projected vector, oriented frame, and unordered subspace are related but not equivalent. The document explicitly says span claims need principal angles and oriented-frame claims require rotation-breaking information.

### Combined-Score Diagnosis

**P1.** DOC A's combined score depends on A2: normalized entropy of visible samples approximates normalized entropy of the full matrix response.

**P2.** DOC B says A2 is not reliable on the matrices under study; phi should be read operationally as an inductive bias, not a population entropy estimator.

**P3.** M1: on mixed-tail-sharp block 1, gain is nearly flat but phi de-rewards the projected oracle slot-2 direction relative to the combined winner.

**P4.** M2: once carry exists, the combined slot-2 direction pins to `span(V_state)` while the projected oracle is much less carry-aligned.

**P5.** M3: combined is blind to `A_fut`, so it can overfit a local current half-window direction; adding future-window balance can recover the direction in a controlled block-1/oracle-warm diagnostic.

**IC1.** The original combined baseline fails the new oracle-identification goal in several observed ways.

Dependencies: `D3 + P2 + P3 + P4 + P5 -> IC1`.

**IC2.** The older M1/M2/M3 taxonomy is superseded by a broader reliability reading: visible-window evidence can favor non-oracle directions.

Dependencies: `IC1 + DIAG-05 reported separation -> IC2`.

Leak: `IC2` is supported as a reliability/identifiability unification, but "tail conspiracy" as the single causal label is over-specific (L7).

### S6 Design Argument

**P6.** Combined lacks future evidence and uses phi-style within-window entropy reward.

**P7.** HM3 requires nonzero normalized response in sketch/current/future; a missing source drives the harmonic mean down.

**P8.** Frobenius weighting puts raw responses on a comparable "fraction of available energy" scale.

**P9.** `sk_F2_low` is more stationary than full-prefix Frobenius scaling and avoids block-id shrinkage in `u_sk`.

**P10.** Removing S5's hard gate restores smoothness/reachability when all responses are positive, at the cost of row-cheat vulnerability.

**IC3.** S6 is a coherent heuristic response to the combined-score failures: it emphasizes split-source balance, carry awareness, and smooth reachability.

Dependencies: `P6 + P7 + P8 + P9 + P10 -> IC3`.

**C1.** S6 is the current best value-only score-design line on clean/high-entropy rowspace but remains heuristic and incomplete.

Dependencies: `IC3 + reported S6 table + theory status -> C1`.

Leak: the "high-entropy" wording should not be attached equally to HM3, phi, and relH1 (L6).

### Oracle-Balance To Oracle-Identifiability

**P11.** DIAG-04 found oracle u-imbalance in several S6 failures.

**P12.** AB-03/E2 made oracle balance better in selected cases but moved the actual score winner farther above the oracle.

**P13.** Forcing only oracle slot 2 does not recover most matrices; forcing the oracle frame does.

**IC4.** Better oracle balance is not sufficient for recovery; candidate scores must test oracle-vs-winner gaps, not only audit the oracle's features.

Dependencies: `P11 + P12 -> IC4`.

**IC5.** Slot-2-only reasoning is insufficient when slot-1/frame anchoring is wrong.

Dependencies: `P13 -> IC5`.

**C2.** The controlling criterion should be oracle identifiability at vector and frame levels, with screens S-1 through S-5.

Dependencies: `IC4 + IC5 + D5 + D6 -> C2`.

This is one of the strongest argument chains in the document.

### Plateau / Rank-r Lift Argument

**P14.** S6 optimizer scores above `oracle_v2_proj` on mixed-tail-sharp but has low alignment.

**P15.** Plateau-width and greedy-vs-joint diagnostics show implemented routines can find non-oracle frames that beat the oracle; the document admits they do not certify the global maximum.

**IC6.** The current score/evidence model does not pin the oracle in the tested landscapes.

Dependencies: `P14 + P15 -> IC6`.

**P16.** A rank-r frame score would be Grassmann-invariant when based on `||A_X Z||_F²`.

**IC7.** Frame-level scoring is the right diagnostic domain for rank-r shipping variants.

Dependencies: `D6 + P16 -> IC7`.

**C3.** FAM-01 rank-r lift should be the next structural step before more scalar epicycles.

Dependencies claimed: `IC6 + IC7 + priority logic -> C3`.

Audit status: reasonable as a project priority, but the geometric plateau-shrink premise is not established (L1, L2).

### Regime / Complementarity Argument

**P17.** iSVD/SVD naturally catches low-sample-entropy directions.

**P18.** The score family is intended to cover high-sample-entropy or stable-spread directions.

**P19.** S6 is not yet competent on several matrices classified as high-sample-entropy/home turf.

**IC8.** The immediate priority is fixing the high-entropy/stable-spread regime; low-entropy spiky-residual gains are downstream.

Dependencies: `P17 + P18 + P19 -> IC8`.

**C4.** FAM-01, Q13, and reliability-aware evidence upgrades should be judged primarily on high-entropy/home-turf cases.

Dependencies: `IC8 + Q10/DIAG-01 labels -> C4`.

Leak: the regime labels need per-slot/per-block precision to support this load (L5).

### Evidence-Model Expansion

**P20.** Current `u_X(v)` variables are magnitude-only proxies.

**P21.** Magnitude alone does not establish support, consistency, robustness, or perturbation stability.

**P22.** HM3 enforces only magnitude balance; it cannot detect incompatible row structures across windows.

**IC9.** A complete evidence model needs reliability-aware `u_X` definitions or additional terms: support, cross-window response consistency, robust aggregation, and subsampling stability.

Dependencies: `P20 + P21 + P22 -> IC9`.

**C5.** After rank-r scoring, the next evidence upgrades should modify reliability of the evidence, not merely add more scalar aggregators over the same magnitude-only terms.

Dependencies: `IC9 + priority order -> C5`.

This chain is logically clean and usefully limits the earlier HM3 story.

### Operational Success Criterion

**P23.** The implicit baseline to beat is value-only online, not `combined`.

**P24.** The displayed online numbers are oracle-aware because the candidate pool includes a projected exact slot-2 direction.

**IC10.** The displayed online table is a ceiling, not the operational target.

Dependencies: `P23 + P24 -> IC10`.

**C6.** Each score-design proposal should be evaluated on the §6 matrix table against the value-only online rerun once available.

Dependencies: `IC10 + Q0 -> C6`.

Leak: before INFRA-10 values are propagated, the document cannot quantify the actual target gap (L4).

## Additional Flagged Leaks

### L8. "Frobenius weighting has no reasonable alternative" is contradicted by later open alternatives

Severity: Moderate  
Confidence: High  
Scope: Sections 2, 2bis, 1quater/Q16

Problem: Section 2bis calls rank-r carry over full-prefix Frobenius "well-justified" with no reasonable alternative, while the same document keeps operator norm, per-vector spectral weighting, leverage weighting, and carry-confidence multipliers alive as possible alternatives to the broader weighting scheme. The likely intended claim is narrower: `sk_F2_low` is better than `sk_F2_full` among those two denominators.

Tighter version: "The full-prefix denominator is refuted for `u_sk`; broader normalization and weighting choices remain open."

### L9. Block-1 HM2 "right form" overstates a post-hoc equivalence

Severity: Moderate  
Confidence: Medium  
Scope: Section 4

Problem: Section 4 says HM2 is the "natural" and "right" block-1 form because it matches c-weighted HM-evi up to scale. That shows equivalence to another chosen objective, not independent optimality. It also shows why the projected oracle can lose at b1.

Tighter version: "HM2 is the internally consistent block-1 fall-through for the S6 family and matches the earlier c-weighted diagnostic under the current row-energy conditions."

### L10. Row-cheat dominance is a necessary screen, but not sufficient for reliable evidence

Severity: Moderate  
Confidence: High  
Scope: S-3, 1ter tail-conspiracy discussion, 6b

Problem: S-3 says the score must rank oracle above row-cheat. That is a valid regression screen against an extreme failure. But the text sometimes lets "oracle >= row-cheat" stand in for tail-conspiracy control. A candidate can pass the extreme row-cheat screen and still be dominated by multi-row tail conspiracies, incompatible cross-window structures, or high-variance evidence.

Tighter version: "S-3 rejects the extreme row-of-window cheat; it does not certify reliability."

### L11. "Oracle-identifying iff argmax coincides with oracle" needs uniqueness/tolerance handling

Severity: Moderate  
Confidence: Medium  
Scope: 1quinquies, S-1/S-2

Problem: The definition says a score is oracle-identifying iff its argmax coincides with the oracle target. In symmetric or repeated-singular-value cases, the oracle may not be unique; in numerical tests, near ties matter. The screens include "near-zero within tol" for S-1, but the definition itself is exact and uniqueness-flavored.

Tighter version: "Oracle-identifying means the oracle target is in the top score-equivalence class within tolerance, with span-level equivalence used when the true target is non-unique."

### L12. Vector, frame, and span targets are mostly separated, but some conclusions still slide between them

Severity: Moderate  
Confidence: Medium-High  
Scope: Sections 1quinquies, 3, 5, Q17

Problem: The document explicitly warns not to interchange slotwise vectors, oriented frames, and unordered subspaces. Still, some priority claims use "slot-2," "frame," and "subspace" evidence as if they diagnose the same failure. Example: force-oracle-frame recovery supports a joint-frame issue, while S6 vector-vs-oracle gaps support a slotwise score issue.

Tighter version: attach each claim to one target: vector S-1, oriented frame S-2, row-cheat frame S-3, or terminal subspace cos².

### L13. "M1 is the bias firing correctly" conflicts with using M1 as a combined-score failure unless the goal is restated

Severity: Moderate  
Confidence: High  
Scope: Sections 1bis, 1ter

Problem: The document says M1 explains combined failure, then later says M1 is the high-entropy bias firing correctly because the oracle slot-2 is concentrated. Both can be true under different goals: oracle recovery vs complementarity with iSVD. Without restating the goal at the transition, the same event is classified as both bug and intended behavior.

Tighter version: "M1 is a failure relative to standalone oracle recovery, but expected behavior under the later SVD-complementarity policy."

### L14. "S6 wins/ties" statements sometimes compress slot-1 and slot-2 outcomes

Severity: Moderate  
Confidence: Medium  
Scope: Section 6

Problem: The table reports cos0² and cos1², but prose conclusions sometimes summarize by matrix. Since the central claim is slot-2 difficulty, matrix-level "wins" or "ties" can hide slot-specific regressions or slot swaps.

Tighter version: report conclusions separately for slot 0, slot 1, and summed/subspace capture.

### L15. Value-only constraint and future peek need scope clarification

Severity: Minor  
Confidence: Medium  
Scope: Sections 1, 2, Q0

Problem: The score is called value-only while using `A_fut`, a peek half-window. This is likely allowed by the benchmark design, but "online" and "value-only" can be read as no future access. The document distinguishes oracle-free from oracle-aware more clearly than it distinguishes streaming-causal from peek-window.

Tighter version: define value-only as "no `V_exact` or oracle-derived candidate," not "causal/no future rows."

### L16. "No reasonable alternative" / "right structural response" language is stronger than the experiment queue

Severity: Minor  
Confidence: High  
Scope: Sections 2bis, 7

Problem: The overview is strongest when it treats FAM-01, Q13, and reliability-aware `u_X` changes as falsifiable next tests. Occasional wording turns them into conclusions before the experiments land.

Tighter version: keep "indicated," "motivated," or "next diagnostic" until the T2/T3 validation exists.

### L17. Combined derivation status is clean, but should not transfer theorem-shaped authority to S6

Severity: Minor  
Confidence: High  
Scope: Theory status, Sections 1ter/2

Problem: The document says this correctly: combined is derived under A1/A2; S6 is a heuristic max-min stability surrogate. The risk is rhetorical carryover from "replacement for combined" to "same theoretical footing." The current text mostly avoids that, so this is a watch item rather than a major leak.

Tighter version: whenever comparing combined and S6, repeat that only combined has the DOC A lower-bound derivation.

## Summary Verdict

The strongest current argument is:

`combined assumptions fail empirically -> S6 is a coherent split-source stability heuristic -> oracle balance alone is insufficient -> oracle-vs-winner frame screens are mandatory -> evidence must become reliability-aware, not just magnitude-balanced.`

The weakest load-bearing moves are:

1. treating generic scalar level-set geometry as proof of the plateau mechanism;
2. treating rank-r lift as geometrically plateau-shrinking before showing it;
3. saying slot-1 is solved while later force-oracle evidence says frame anchoring is broken;
4. using oracle-aware online numbers as more than a ceiling;
5. using "high-sample-entropy" as a matrix-level label when the logic needs slot/block-level labels.

None of these require abandoning the score-design line. They mainly require tightening the claim scopes so the next experiments falsify the actual premises rather than a softened narrative version of them.
