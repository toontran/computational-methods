# Methodology review - score-design experiment shape

## Top flags

- **Major | High confidence | Scope: operational baseline and ship criterion.**
  The design's stated baseline is moving from oracle-aware `future_hmean_online`
  to value-only online, but the comparison table and several success statements
  still depend on the oracle-aware ceiling. Until the value-only rerun is
  propagated into the main acceptance table, the experiment cannot cleanly answer
  "does score-design close the operational gap?" It can only answer "how far is
  S6 from an upper-bound policy whose pool contains the answer."

- **Major | High confidence | Scope: central causal claim.**
  The current suite mixes several failure hypotheses: wrong score argmax,
  greedy slot-1 anchoring, rank-1 plateau drift, mature-carry distortion, and
  missing reliability evidence. The canonical screens S-1 through S-5 are the
  right decomposition, but they are not yet all part of one required factorial
  experiment. Without that, a T3 failure cannot be attributed uniquely to
  feature insufficiency, optimizer/landscape failure, or downstream streaming
  state corruption.

- **Major | Medium-high confidence | Scope: generality.**
  Most decisive diagnostics are concentrated on a small, hand-curated matrix
  set, selected blocks such as b31, and rank-2/slot-2 behavior. That is good for
  debugging, but underdetermines claims about rank-r score design unless the
  same controls are repeated across rank, block time, window size, and matrix
  regimes.

## Design strengths

- **Moderate | High confidence | Scope: question formulation.**
  The reframing from "does the oracle have balanced u-values?" to "is the
  oracle frame the score maximizer?" is methodologically strong. It makes the
  experiment test the decision-relevant property of the score rather than an
  attractive but non-decisive audit metric.

- **Moderate | High confidence | Scope: baseline hygiene.**
  The overview explicitly identifies the oracle leak in the prior online
  baseline and demotes those numbers to a ceiling. This prevents a value-only
  score family from being judged against a non-deployable comparator, provided
  the corrected value-only table becomes the actual ship criterion.

- **Moderate | High confidence | Scope: effect isolation.**
  The proposed S-1/S-2 oracle-vs-winner screens, S-3 row-cheat screen, S-4
  force-oracle ceilings, and S-5 anchor-sensitivity checks isolate distinct
  mechanisms in principle. Together they can separate "oracle unreachable,"
  "score peak wrong," "row cheat wins," "slot-1 anchor bad," and "joint rank-r
  search needed."

- **Moderate | Medium-high confidence | Scope: success metric.**
  Moving the rank-r diagnostic to principal angles and Grassmann-invariant
  frame scores is the right metric design. It avoids over-reading oriented
  slotwise cosines when the operational object is a retained subspace.

- **Minor | High confidence | Scope: negative results.**
  The document treats failed variants such as E2, GM, and D0 as falsifying
  specific design principles rather than as global refutations. That is a good
  experimental habit and keeps the search from overgeneralizing one ablation.

## Design gaps

- **Major | High confidence | Scope: fixed vs varied conditions.**
  The overview is clear about many fixed ingredients in individual diagnostics
  but does not yet present a single matrix of what is held fixed across S6,
  FAM-01, AB-03, relH1 variants, online, and iSVD. The key fixed factors should
  be matrix generator, seed, block, rank, half-window, future peek length,
  optimizer budget, restart policy, carry initialization, and downstream SVD
  update. Without that matrix, readers cannot tell whether each comparison is
  a one-factor ablation or a bundle of changes.

- **Major | High confidence | Scope: ablation structure.**
  Some ablations are one-at-a-time, but the broader progression is not fully
  factorial. For example, rank-r lift, carry-confidence weighting, robust row
  aggregation, cross-window correlation, and relH1-style support controls each
  target different mechanisms. Running them sequentially is efficient, but it
  risks confounding a negative result with the absence of a needed companion
  control. At minimum, the design needs a small pre-registered interaction grid
  after each single-factor screen passes.

- **Major | Medium confidence | Scope: score-family sufficiency.**
  The current evidence variables are mostly magnitude-only `u_X` responses.
  If HM3, E2, GM, or D0 fail, that falsifies those forms, not the sufficiency
  of `{A_sketch, A_cur, A_fut}` as available data. To claim the evidence model
  is incomplete, the experiment needs controls over alternative functions of
  the same sources: additive/log-additive forms, min/max forms, cross-source
  response alignment, robust row-energy aggregation, and simple Gram/power-step
  proxies.

- **Moderate | High confidence | Scope: operational metric.**
  The document uses several metrics: score gap, oracle-vs-winner ordering,
  principal angles, slotwise cos², capture out of 2, and terminal T3 final
  cos². These are all useful, but the design does not yet define a metric-link
  rule. If score-gap improvements do not predict principal-angle improvement
  or terminal streaming capture, then oracle argmax is an incomplete success
  metric for the operational goal.

- **Moderate | Medium-high confidence | Scope: lookahead design.**
  `A_fut` is central to S6, but the lookahead condition itself is not treated
  as an experimental factor. Varying future-window size, offset, overlap, and
  independence would distinguish "future evidence is necessary" from "this
  particular peek horizon and HM3 normalization happen to work or fail."

- **Moderate | Medium confidence | Scope: block-time effects.**
  Many mechanistic statements depend on b31 mature-carry behavior, while b1/b2
  diagnostics have different information structure. The experiment should make
  early, middle, and late blocks a standard crossed factor. Otherwise late
  sketch accumulation, block-1 fall-through, and score design are partially
  confounded.

- **Moderate | Medium confidence | Scope: regime labels.**
  The HIGH/BOUNDARY/LOW sample-entropy split is useful, and DIAG-01 supports
  it, but the ship criterion still needs regime-balanced sampling. A high-
  entropy "home turf" claim should not rely mainly on the same counterexample
  matrices that motivated the design.

## Missing controls

- **Major | High confidence | Scope: required controls.**
  Make S-1 through S-5 mandatory for every new score proposal before T3
  interpretation: vector oracle-vs-winner, frame oracle-vs-winner, row-cheat
  dominance, force-oracle-v2/frame ceilings, and greedy-vs-oracle-anchor
  sensitivity. This turns the diagnostic list into an actual experimental
  design.

- **Major | High confidence | Scope: value-only baseline.**
  Replace all operational online comparisons with the value-only rerun and keep
  the oracle-aware online result only as a labeled ceiling. A score family
  should not be accepted or rejected against a comparator that contains
  `V_exact`.

- **Major | Medium-high confidence | Scope: source isolation.**
  Add source ablations under the same optimizer and metric suite:
  `{sk}`, `{cur}`, `{fut}`, `{sk,cur}`, `{sk,fut}`, `{cur,fut}`,
  `{sk,cur,fut}`. This isolates which visible source supplies information and
  which source introduces misleading evidence.

- **Moderate | Medium-high confidence | Scope: rank-r claim.**
  Run the frame-level audit at a small rank ladder, for example r=2, r=3, r=4,
  while holding matrix, blocks, windows, and optimizer budget fixed. If rank
  alone fixes the plateau, the design question changes from score features to
  representation capacity.

- **Moderate | Medium-high confidence | Scope: optimizer reliability.**
  Standardize high-budget restarts, oracle warm starts, state warm starts, and
  independent random seeds for each nonconvex score. Otherwise "winner beats
  oracle" may reflect optimizer coverage rather than the true landscape.

- **Moderate | Medium confidence | Scope: window/lookahead.**
  Cross half-window length and future-peek length on a small matrix subset.
  This is the cheapest way to test whether failures are sample-size artifacts
  rather than score-design artifacts.

- **Moderate | Medium confidence | Scope: reliability evidence.**
  Add row-support, robust-energy, and cross-window response-correlation controls
  as evidence definitions, not only as post-hoc diagnostics. The current
  magnitude-only `u_X` variables cannot in principle distinguish stable spread
  from tail-row coincidence.

- **Minor | Medium confidence | Scope: positive/negative controls.**
  Include synthetic matrices where the oracle is provably identifiable from
  one source alone, and matrices where visible split-window energies are
  symmetric but the global oracle is intentionally ambiguous. These check that
  the audit can both pass and fail for the right reason.

## Falsification experiment

**Major | High confidence | Scope: minimal decisive experiment.**

Run one pre-registered crossed diagnostic on the same value-only code path:

1. Choose a small matrix panel: at least two high-entropy home-turf failures,
   one home-turf success/control, one boundary case, and one low-entropy SVD
   territory case.
2. For each matrix, test early/mid/late blocks and r in `{2, 3, 4}` with fixed
   half-window, future peek, optimizer budget, seeds, and carry initialization.
3. For each condition, evaluate four feature classes:
   `E0 = magnitude-only {u_sk,u_cur,u_fut}`;
   `E1 = E0 + source ablations and weighted HM/min/log-additive variants`;
   `E2 = E1 + reliability evidence`, such as per-source relH1, robust
   row-energy, or capped row influence;
   `E3 = E2 + cross-window response-consistency` terms.
4. For every score, require S-1 through S-5 before interpreting T3. Report
   score gap, row-cheat gap, principal angles to the projected oracle frame,
   and terminal streaming cos²/capture.

This experiment falsifies the current priority hypothesis if the rank-r lift
alone does not improve S-2/S-5 or T3 on the high-entropy home-turf cases. It
falsifies the "magnitude-only visible evidence is enough" hypothesis if only
`E2` or `E3` makes the oracle frame the winner. It falsifies the operational
success metric if oracle-vs-winner and principal-angle improvements do not
translate into terminal streaming capture under the same downstream pipeline.
