# S7 Pareto override — making S7 ≥ combined

Date: 2026-05-02

## Problem

S7's wiring (`half_window_sliding_hmean_experiment.py:786-796`) replaces
`V_default[:,1]` (combined's joint rank-2 slot-2) with
`rank2_svd_frame(V_default[:,0], chosen_v2_S7, M_gain)`. On matrices
where combined's joint slot-2 is structurally better than S7's deflated
greedy slot-2 (e.g. diffuse-diffuse, mixed-tail-soft), S7 underperforms
combined despite seeing more evidence (peek + entropy on stacked rows).

## Fix

Add a `--rsk-pareto-metric` knob:

- `off` (default): current behavior. V_selected = rank2_svd_frame(V_default[:,0], chosen_v2_S7).
- `s7frame`: pick whichever of {V_default[:,0:2], rank2_svd_frame(V_default[:,0], chosen_v2_S7)} maximizes the S7 frame score.
- `mgain`: pick whichever maximizes ||M_gain V||_F² (combined's objective).
- `always-default`: V_selected = V_default[:,0:2] (S7 ≡ combined for slot selection — sanity control).

S7 frame score (`s7_frame_score` in the experiment file):
`(||A_sk V||_F² + ||A_cur V||_F² + ||A_fut V||_F²) · relH1(stacked row energies of [A_sk V; A_cur V; A_fut V] summed across columns of V)`. Grassmann-invariant.

## Results (cos0² / cos1², half_win=32, seed=0)

| matrix              | combined    | S7 old      | S7 pareto-s7frame | S7 pareto-mgain |
|---------------------|-------------|-------------|-------------------|-----------------|
| diffuse-diffuse     | 0.483/0.248 | 0.595/0.040 | 0.595/0.040       | **0.708/0.242** |
| mixed-tail-sharp    | 0.894/0.019 | 0.778/0.028 | 0.819/0.059       | 0.838/0.027     |
| mixed-tail-balanced | 0.756/0.075 | 0.907/0.111 | 0.907/0.111       | **0.910/0.605** |
| mixed-tail-soft     | 0.886/0.304 | 0.890/0.002 | 0.890/0.002       | **0.903/0.755** |
| static-cex          | 0.975/0.912 | 0.983/0.019 | 0.983/0.019       | 0.974/0.892     |

`always-default` on diffuse-diffuse: 0.483/0.248 — matches combined exactly (sanity).

## Pareto-mgain wins vs combined

- diffuse-diffuse: cos0² +0.225, cos1² tied
- mixed-tail-balanced: cos0² +0.154, **cos1² +0.530**
- mixed-tail-soft: cos0² +0.017, **cos1² +0.451**
- static-cex: matches within noise
- mixed-tail-sharp: minor cos0² drop, cos1² tied

## Mechanism

`rank2_svd_frame(V_default[:,0], chosen_v2, M_gain)` is by construction
the M_gain F²-maximal rank-2 frame inside span(V_default[:,0], chosen_v2).
When chosen_v2_S7 enriches this 2D plane with a high-M_gain-energy
direction, V_svd has higher M_gain F² than V_default[:,0:2]. When
chosen_v2_S7 is the score-design "spread" direction (high entropy bias,
low M_gain energy), V_default[:,0:2] wins.

The `mgain` metric thus uses S7's optimizer as a basin-enrichment proposer
and combined's objective as the arbiter. On matrices where S7's score-
favoured direction happens to ALSO have high M_gain energy, Pareto-mgain
takes it; otherwise it falls back to combined's joint solution.

## Why s7frame doesn't work as well

S7's frame score genuinely ranks chosen_v2_S7 higher than V_default[:,1]
on diffuse-diffuse / mts-soft (the score's argmax is off-oracle, §1quinquies
S-1 failure). Picking by S7 frame score keeps the optimizer's chosen v —
which is the same as no override.

## Caveat

- Single seed, 5 matrices. mts-sharp regresses cos0² by −0.056 — needs
  cross-seed verification.
- `mgain` Pareto is using combined's objective as an arbiter, so this
  variant is structurally combined-with-S7-as-proposer rather than a pure
  S7 family member. Whether to label this S7-mgain or as a new S9-style
  variant is a design call.

## Files

- `summary/score_family_pareto_override/{matrix}_S7_{pareto,mgain,always_default}.{json,csv,txt}`
- Code: `half_window_sliding_hmean_experiment.py` lines 279-300 (s7_frame_score),
  ~786-830 (Pareto override), CLI flag `--rsk-pareto-metric`.
