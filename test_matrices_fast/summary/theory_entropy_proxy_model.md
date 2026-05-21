TH-02 entropy-proxy model study
===============================

Date: 2026-04-28
Backlog item: score_family_workflow.txt [TH-02]
Resolves: score_design_overview.txt Q9 at the model-assumption level.


1. Verdict
----------

Verdict: SHIP as a model-assumption note; ITERATE empirically.

The analytic models below satisfy TH-02's acceptance bar: they give simple
conditions under which high visible-window response entropy is evidence for
high population response entropy of oracle directions, and conditions under
which a finite visible sample can be fooled by tail conspiracy.

This does NOT make S6 / HM3 / relH1 theorem-backed. The score family remains a
heuristic unless TH-01 / TH-02 / TH-03 together justify a precise, limited
claim and DIAG-01/DIAG-02 measure the actual matrices. The result here is
narrower:

  Visible-window entropy is a useful proxy only under row-exchangeability,
  bounded leverage or stable mixture weights, and enough effective samples in
  each active component. It is unreliable under rare-tail mixtures whose
  high-energy rows can appear in A_cur and A_fut but not represent stable
  population support.

No new diagnostic was added, so no diagnostic_toolkit propagation is needed.


2. Quantities
-------------

For a row distribution P over rows a and a unit direction v, define the
population response weights by

  Z_v = (a^T v)^2,        mu_v = E_P[Z_v].

The population entropy of v means the entropy / effective support of the
normalized contribution distribution over independent rows:

  p_i(v) = Z_i / sum_j Z_j
  H(v) = -sum_i p_i log p_i
  N_eff(v) = exp(H(v)).

For a finite window W, H_W(v) and N_eff,W(v) are the same quantities restricted
to rows in W. relH1 in the diagnostics is a related within-window concentration
flag: high relH1 means many rows contribute, low relH1 means a few rows carry
the response. HM3 is the between-window proxy: it rewards v only when normalized
energy is present in sketch, current, and future windows.

Proxy validity means:

  H_visible(v_oracle) high in A_cur/A_fut/sketch predicts H_population(v_oracle)
  high, and high-scoring non-oracle directions with visible high entropy keep
  similar support on the next unseen window.

Proxy failure means:

  visible windows show balanced or high entropy, but the apparent support is
  caused by finite-sample rare rows, unstable mixture components, or low-rank
  spikes that do not replicate as population support.


3. Model A: diffuse exchangeable rowspace
-----------------------------------------

Model:

  Rows are exchangeable draws

    a = Sigma^(1/2) x,     x subgaussian, isotropic, ||x||_{\psi_2} bounded,

  with no row atom carrying more than O(log n / n) of the response mass for the
  oracle directions. Equivalently, for v in the candidate/oracle set,
  Z_v=(a^T v)^2 has bounded coefficient of variation or a subexponential tail
  with stable mean.

Characterization:

  For a fixed v, empirical energy and empirical entropy concentrate once the
  window size n is larger than the effective support scale. Uniformly over a
  low-dimensional candidate span, an epsilon-net argument gives the same story:
  if no row has large leverage, high H_W(v) is strong evidence that the
  population response is diffuse.

What the score axes mean:

  - HM3 is appropriate because balanced normalized energy across sketch,
    A_cur, and A_fut is a split-sample stability check.
  - relH1 is secondary; bounded leverage already makes within-window
    concentration unlikely.
  - rank-r lift helps when the top-r oracle space is a subspace rather than a
    unique vector, because scalar slot-2 scoring can drift along a flat
    high-entropy ridge even when the entropy proxy itself is valid.

Expected matrices:

  diffuse-diffuse and etf-basket-basis are the cleanest matches. mixed-tail-soft
  may be close if its tail component is weak enough that no rare row dominates
  oracle slot-2.


4. Model B: heavy-tail mixture / tail conspiracy
------------------------------------------------

Model:

  Rows come from a mixture

    a ~ (1 - pi) P_bulk + pi P_tail,

  where P_bulk is diffuse but P_tail has high norm or high alignment with a
  small set of right-space directions. The visible half-windows contain only
  n*pi tail rows in expectation. In the dangerous regime, n*pi is O(1) to
  O(log n), and tail row magnitudes are large enough that a few tail rows
  dominate Z_v.

Tail-conspiracy failure:

  A non-oracle v can score well if A_cur and A_fut both happen to contain tail
  rows aligned with v. HM3 sees between-window balance and treats it as stable,
  but population support is still low: most of the expected response comes from
  rare events, not from a broad row population. If the same tail pattern does
  not appear in the next unseen window, the carried direction decays.

Characterization:

  The proxy is valid only if either pi*n is large enough that each visible
  window samples the tail mixture accurately, or tail magnitudes are clipped /
  leverage-controlled so no few rows dominate the normalized response. It fails
  when rare rows produce balanced visible energy in both half-windows while
  within-window entropy remains low.

What the score axes mean:

  - HM3 catches the easy failure where the tail appears in only one visible
    half-window.
  - relH1 is the direct guard for the harder failure where both visible windows
    contain a few aligned tail rows.
  - rank-r lift helps if the correct object is a tail-plus-bulk subspace, but it
    does not by itself distinguish stable mixture support from rare-row support.

Expected matrices:

  mixed-tail-sharp, mixed-tail-balanced, and static-cex are the likely boundary
  or failure cases. mixed-tail-soft should fail less often because the tail is
  weaker.


5. Model C: spiky low-entropy residual
--------------------------------------

Model:

  Rows contain a low-rank or sparse shock component

    a = b + s e_j,

  where b is diffuse background and the shock term is concentrated on a small
  number of rows or row types. The oracle slot may be a genuine low-entropy
  direction: population energy is correctly concentrated, not a sampling
  artifact.

Characterization:

  Here high visible entropy is not expected for the oracle residual direction,
  and lack of high entropy is not a bug in the matrix. A score family designed
  as the high-entropy complement to iSVD should not use this regime as its
  primary ship criterion. The entropy proxy is not "broken"; it is simply aimed
  at the wrong part of the spectrum.

What the score axes mean:

  - HM3 may reject the shock direction if it is absent from one visible window.
  - relH1 may reject it even when the shock is stable, because the population
    support is genuinely low.
  - rank-r lift can carry both diffuse and spiky components if the frame score
    is allowed to keep low-entropy directions through an SVD/hybrid branch.

Expected matrices:

  residual-spiky-shocks and risk-residual-panel fit this model. They should be
  treated as complementarity tests against iSVD, not as evidence that the
  high-entropy score family failed on its intended home turf.


6. Model D: coherent dictionary / ETF basket
--------------------------------------------

Model:

  Rows are sampled from a finite set of coherent but bounded-norm atoms

    a in {d_1, ..., d_m},     P(a=d_j)=w_j,

  with many active atoms per oracle direction and no extreme tail magnitude.
  Coherence creates multiple near-equivalent right directions, but row leverage
  is bounded.

Characterization:

  Visible entropy predicts population entropy if each active atom has expected
  count n*w_j large enough in the visible windows. Failure is less about rare
  tails and more about identifiability: many directions can have similar
  high-entropy response, so scalar scores can plateau even when the proxy is
  valid.

What the score axes mean:

  - HM3 is a useful split-sample stability check.
  - relH1 is mostly a sanity guard against atom imbalance.
  - rank-r lift is the main structural fix because the object is naturally a
    subspace/frame, not a single slot-2 vector.

Expected matrices:

  etf-basket-basis is the main match.


7. Matrix-by-matrix hypothesis table
------------------------------------

This is a hypothesis table, not a measured regime audit. DIAG-01 still owns the
empirical HIGH/LOW labels.

| Matrix | Model match | Expected proxy validity | Expected failure mode | Predicted helpful score axis |
| --- | --- | --- | --- | --- |
| diffuse-diffuse | Diffuse exchangeable rowspace | High: visible split windows should represent population rowspace when window samples are adequate. | Scalar plateau / carry drift after early lock-on, not entropy proxy failure. | Rank-r lift first; HM3 already has the right between-window signal; relH1 secondary. |
| etf-basket-basis | Coherent bounded dictionary | High to medium-high: bounded atoms make visible entropy meaningful, but coherent atoms create near-ties. | Identifiability plateau among equivalent high-entropy directions. | Rank-r lift first; HM3 useful; relH1 as imbalance guard. |
| mixed-tail-soft | Weak heavy-tail mixture | Medium: tail is present but likely not dominant enough to fully break split-sample evidence. | Mild rare-row support or slot ambiguity when tail rows align in one window. | HM3 for between-window balance; rank-r lift for ambiguity; relH1 if low support is observed. |
| mixed-tail-balanced | Heavy-tail mixture | Medium-low: visible entropy can be informative only if tail counts are stable in both windows. | Tail conspiracy: both visible windows contain a few aligned tail rows that do not replicate forward. | relH1 for within-window support plus rank-r lift; HM3 catches only one-sided tail imbalance. |
| mixed-tail-sharp | Strong heavy-tail mixture | Low for slot-2 unless tail support is measured stable. | Severe tail conspiracy and wide scalar plateau; oracle slot may itself be boundary low-entropy. | relH1 tail guard and rank-r lift; HM3 alone insufficient. |
| static-cex | Structured counterexample / tail mixture | Low-medium: proxy may be invalid if constructed rows create finite-window support that is not population-stable. | Adversarial or near-adversarial visible support; score can prefer balanced non-oracle directions. | rank-r lift for subspace structure; relH1 if support is row-concentrated; HM3 only catches between-window imbalance. |
| residual-spiky-shocks | Spiky low-entropy residual | Low by design for the spiky oracle direction; this is iSVD territory. | The score rejects a genuinely low-entropy population direction, which is expected under complementarity. | Hybrid/SVD branch or rank-r frame that can carry low-entropy directions; relH1 is not a primary fix. |
| risk-residual-panel | Spiky low-entropy residual | Low by design for residual shock directions. | Stable but low-support residual factors look bad to high-entropy scores. | Hybrid/SVD branch; rank-r lift only if combined with low-entropy allowance. |


8. Assumptions and limits
-------------------------

Assumptions under which visible entropy is a valid proxy:

  - Rows in visible and future windows are exchangeable or drawn from a stable
    mixture with stationary weights.
  - No single row or O(1) set of rows can dominate normalized response for the
    oracle high-entropy directions.
  - Each active mixture component has enough expected visible samples to make
    H_W(v) a stable estimate.
  - Candidate search is restricted to the union span / low-dimensional frame
    being scored; this note does not claim uniform concentration over the full
    ambient sphere.

Assumptions under which the proxy fails:

  - Rare tail components have high magnitude and O(1) visible counts.
  - A_cur and A_fut are not independent checks because the construction couples
    their tail rows or repeats row types.
  - The population oracle direction is genuinely low-entropy.
  - The score is scalar-per-v and the correct target is a rank-r subspace with a
    wide set of equivalent high-entropy directions.

Implication for the score family:

  HM3 is best read as a split-sample balance heuristic. relH1 is the natural
  within-window support guard. The rank-r lift is the natural fix for subspace
  identifiability and scalar plateau drift. None of these axes alone proves
  that visible-window entropy is population entropy.


9. Propagation notes
--------------------

Updated:

  - score_design_overview.txt Q9: mark model-assumption resolution with pointer.
  - score_family_workflow.txt TH-02: mark done and add sequencing note.

Not updated:

  - diagnostic_toolkit.txt: no new diagnostic or closed diagnostic gap.
  - score_design_overview.txt measured regime labels: DIAG-01 owns those.
  - HM3 theory-status language: TH-01 owns bound/surrogate/heuristic status.

Orchestrator integration note:

  If DIAG-01 finds that a matrix currently called HIGH has low oracle response
  entropy, reclassify it as a boundary case. This TH-02 note predicts
  mixed-tail-sharp and static-cex are the most likely candidates for that
  correction, but does not measure them.
