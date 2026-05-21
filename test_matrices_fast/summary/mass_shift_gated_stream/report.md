# Mass-shift-gated streaming policy — results

Date: 2026-05-07
Probe code: `mass_shift_gated_stream.py`
Cells: `results.csv` (96 rows = 4 matrices × 3 seeds × 8 policies)

## Question

The v_cur-vs-v_fut probe (`summary/per_block_v_cur_vs_v_fut/`) showed that
`mass_shift = |‖A_cur·v_cur‖² − ‖A_cur·v_fut‖²| / max(...)` discriminates
regimes (median 0.33 on diffuse-diffuse vs 0.11 on static-cex).

Question: can mass_shift be used at runtime to select per-block between

  - 2-way: {combined, iSVD}
  - 3-way: {combined, iSVD, oracle-projection}

so that the gated stream beats both pure baselines?

## Setup

At each block t, with the current gated state S_t:
1. Run combined optimizer to get v_combined (slot-1).
2. Run combined optimizer with extended block [A_cur; A_fut] to get v_fut.
3. Compute mass_shift from (e_cur, e_fut) on A_cur.
4. Pick slot-1 by policy:
     2-way: iSVD if mass_shift ≥ τ else combined.
     3-way: oracle if mass_shift ≥ τ_oracle; iSVD if mass_shift ≥ τ_isvd; else combined.
5. Slot-2 always from combined optimizer (orthogonalized against slot-1).
6. make_state advances S_{t+1}.

Pure baselines:
  pure_combined — slot-1 and slot-2 from combined optimizer.
  pure_isvd     — top-2 right SVs of M_gain = [B_top; A_cur] (matches bench
                  `policy == "isvd"`).
  pure_oracle   — V_exact[:, 0:2] projected onto rowspace(M_gain),
                  orthogonalized.

half_win=32, n=1024, rank=2, 16 blocks/seed, 3 seeds, 4 matrices, K=16
ceiling cos² = K·δ ≈ 0.59.

## Headline: mean final cos²(state.V[:,0], V_exact[:,0])

```
matrix             pure_comb  pure_isvd  pure_oracle  g2_τ=.15  g2_τ=.20  g2_τ=.30  g3_.15_.35  g3_.20_.40
static-cex            .451       .001       .428        .001      .001      .001      .030       .107
mixed-tail-sharp      .087       .001       .405        .001      .001      .003      .025       .013
mixed-tail-soft       .207       .014       .386        .010      .004      .056      .342       .193
diffuse-diffuse       .095       .113       .384        .113      .077      .082      .463       .503
```

(Median rows in `results.csv`; pattern matches mean.)

## What the pure baselines tell us

The pattern matches visibility-analysis §5 qualitatively, modulo our
K=16 horizon:

- **static-cex (row-concentrated)**: pure_combined (.451) wins;
  pure_isvd ≈ 0.
- **diffuse-diffuse (row-diffuse)**: pure_isvd (.113) > pure_combined (.095).
- **mts-sharp / mts-soft (mixed-tail)**: pure_combined > pure_isvd; both
  far below pure_oracle (~.40).

Pure_oracle hits 0.38–0.43 on every matrix — that's our K=16 ceiling
benchmark. Anything ≥ 0.40 is "near-ceiling".

## 2-way gated (combined / iSVD): fails on every matrix

```
matrix             pure_combined  pure_isvd  best 2-way     verdict
static-cex            .451          .001       .001         LOSES vs combined
mixed-tail-sharp      .087          .001       .003         LOSES vs combined
mixed-tail-soft       .207          .014       .056         LOSES vs combined
diffuse-diffuse       .095          .113       .113         MATCHES iSVD only
```

The 2-way gated policy never beats the better pure baseline and often
catastrophically underperforms (static-cex: combined gets .451, gating
gets .001 — three orders of magnitude worse).

**Mechanism**: per-block choice destroys state-evolution consistency.
On static-cex, combined's slot-1 sequence is a coherent direction stream
that make_state compounds into an oracle-aligned carry; replacing one or
two blocks' picks with iSVD's top-SV-of-M_gain (a different objective)
breaks the coherence. The state's leading direction stops accumulating
oracle mass and can't recover even if subsequent blocks revert to
combined.

This is the same lesson visibility-analysis §4 spelled out: "the score's
real role is sketch shaping... it's the *sequence* of V_score choices,
propagated through make_state, that accumulates oracle. The per-block
argmax is irrelevant." Mid-stream policy switches break the sequence.

## 3-way gated (combined / iSVD / oracle): bimodal — wins big on diffuse, loses big on row-concentrated

```
matrix             pure_combined  pure_oracle  g3_.15_.35  g3_.20_.40   verdict
static-cex            .451          .428         .030        .107        LOSES big (oracle access wasted)
mixed-tail-sharp      .087          .405         .025        .013        LOSES big
mixed-tail-soft       .207          .386         .342        .193        WINS at .15_.35; loses at .20_.40
diffuse-diffuse       .095          .384         .463        .503        WINS — beats pure_oracle
```

- **diffuse-diffuse**: gated3 hits .50, *above* pure_oracle's .38.
  Inspecting choice traces: g3_.20_.40 picks oracle for the first 2-3
  blocks and combined for the rest. So the win is "oracle warm-start +
  combined finishing", not really mass-shift discrimination.
- **mts-soft**: g3_.15_.35 = .342 ≈ pure_oracle's .386. Mass-shift
  drives it to oracle on enough early blocks to anchor the carry.
- **static-cex / mts-sharp**: gated3 catastrophically underperforms
  *even pure_oracle*. Same state-evolution-break problem as 2-way.

So the 3-way gate trades catastrophic regression on row-concentrated
matrices for near-oracle performance on diffuse — a Pareto-bad outcome
(the regime where it helps is also the regime where pure_isvd already
helps; the regime where it hurts is also where combined was working).

## Why both gating strategies fail

The sliding-window state has memory: state.V at block t is a compression
of every V_score choice from blocks 1..t-1. make_state's projected SVD
treats these as a coherent stream — it does not commute with choosing
freely from {combined, iSVD, oracle} per block. In the gated runs, every
switch contaminates the carry with a direction that the next block's
combined/iSVD/oracle source does not anchor on.

Empirically:
- on row-concentrated matrices, combined is producing a stream that
  accumulates oracle alignment slowly but coherently. Any switch costs
  more than it gains.
- on diffuse-diffuse, neither combined nor iSVD is doing much; oracle
  warmstart helps the carry escape the wrong attractor early, but
  *static* oracle (pure_oracle) is sub-optimal because oracle
  projection at every block keeps re-anchoring into a possibly-noisy
  V_exact projection. A few oracle blocks then combined finishes
  better.

The mass_shift signal *does* discriminate regimes, but the natural
response (switch policy) is not a state-aware response. To use
mass_shift productively we'd need either:

1. A *single* policy whose internal direction choice is parameterized by
   mass_shift (e.g., a continuous interpolation between combined and iSVD
   directions), so state evolution is smooth — not a hard switch.
2. mass_shift gating as a *runtime monitor* (warning signal for "this
   lock will move"), not a per-block selector.
3. An oracle warmstart heuristic: use oracle for the first K_warm blocks,
   combined thereafter. This is what g3_.20_.40 effectively does on
   diffuse-diffuse, and it is the only setting that beat pure_oracle.

## Falsifiable conclusions

- **Mass-shift gating between {combined, iSVD} per block: NEGATIVE.**
  Strictly worse than the best pure baseline on every matrix tested.
  Tried τ ∈ {0.15, 0.20, 0.30}; none recover combined's static-cex
  performance.

- **Mass-shift gating between {combined, iSVD, oracle} per block: PARTIAL.**
  Wins on diffuse-diffuse (.50 vs .11 baseline) and matches oracle on
  mts-soft, but catastrophically regresses static-cex (.45 → .03–.11)
  and mts-sharp (oracle would give .40, gating gives .01–.03). Net
  result: Pareto-dominated by always running pure_combined on
  row-concentrated matrices and pure_oracle on diffuse.

- **The "warmstart-then-combined" pattern is interesting and worth a
  separate experiment.** g3_.20_.40 on diffuse-diffuse beats
  pure_oracle by picking oracle only at blocks 1-3. A 1-line policy
  "oracle for K_warm blocks, combined after" might dominate pure_oracle
  on diffuse without help from mass_shift.

## Files

- `results.csv` — 96 rows: matrix, seed, policy, final_cos2,
  cos2_block_trace, mass_shift_trace, choices.
- This report.
