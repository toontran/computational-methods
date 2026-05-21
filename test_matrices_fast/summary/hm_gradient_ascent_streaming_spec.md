# HM-combinations as gradient ascent in the streaming sketch loop — implementation spec

Date: 2026-04-25

This is the handoff spec for replacing the existing HM-combination *rerank*
implementation with a *gradient-ascent* implementation that feeds the
ascent-found direction into the next sketch.

## Why this needs to exist

The current `half_window_sliding_hmean_experiment.run_trajectory` uses HM
combinations only as a scalar score over a fixed candidate pool of size 5–7.
That misuses the HM combinations: a re-ranker over a small pool can only
return one of those 5–7 vectors, none of which is the actual maximizer of the
HM objective. The synthesis-report wins for `future_hmean_nested_online` come
almost entirely from one of those candidates being `q2_vs_q1oracle` (a
projection of `V_exact[:, 1]`), i.e. oracle leakage, not from the HM
combination doing real work.

What the HM combinations were meant to do is define *objectives* that the
streaming optimizer actually maximizes per block, so that the chosen `v2` is
fed into `make_state` and the sketch evolves under that objective's
trajectory. That is what this spec implements.

## The framing this spec assumes

Streaming sketch maintenance is structurally a **noisy power iteration on the
second slot**. Each block produces a `v2` that is a noisy estimate of
`V_exact[:, 1]`; the recursive `make_state` (left-projected SVD of
`M_gain @ V_score`) averages those estimates into the carried 2D subspace
`state["V"]`. For convergence of slot 2:

1. Each block's `v2` must have nonzero `||A_{w+1} v||²` — i.e. nonzero
   "signal" for the new row content. Combined-score's `v_carry` violates this
   (`b ≈ 1.5e-5`), so its iteration provides no update for slot 2 across
   blocks. Empirically `mean_align ≈ 0.045` on `mixed-tail-sharp`.
2. Each block's `v2` should still couple to the carried sketch (nonzero
   `||sketch v||²`), otherwise the iterate is replaced rather than averaged
   and slot 2 wanders. This is what HM combinations like `nested` and
   `triplet` enforce structurally.
3. Per-block alignment with `V_exact[:, 1]` will be at noise floor (1e-3 to
   1e-2) on tail-dominant matrices regardless of objective. That is expected
   and not a failure mode. The win comes over many blocks.

The diagnostic `summary/hmean_combinations_optimizer_diagnostic.txt` confirms
that gradient ascent on `nested`, `triplet`, and `weighted` HM produces
in-window optima with ~0.99 mass inside M_gain rowspace and nonzero A_{w+1}
content. `pairwise` doesn't help (both arguments contain the sketch, so
sketch-only directions max it; structurally similar to combined-score).
Plain future-HM puts 50% mass in S4 (zero sketch coupling) and replaces the
iterate rather than averaging it.

## What to build

A new runner, e.g. `half_window_sliding_hmean_gradient.py`, that mirrors
`half_window_sliding_hmean_experiment.run_trajectory` but replaces the
"compute v1 from combined-score, build candidates, rerank by HM" pattern
with "compute v1 from combined-score, run gradient ascent on the HM
combination to get v2, feed [v1, v2] into make_state".

### Per-block algorithm

```
inputs: A_w (current half), A_{w+1} (next half), state (prev sketch),
        old_row_memory, policy ∈ {triplet, nested, weighted}
outputs: V_selected (n × 2), updated state, updated old_row_memory

1. Build M_gain:
     if state is None: M_gain = A_w
     else:             B_top = state["s"][:, None] * state["V"].T
                       M_gain = vstack(B_top, A_w)
     A_sketch_prior = B_top if state is not None else None

2. Run combined-score optimizer to get V_default, v1 = V_default[:, 0]:
     V_default = entropy_iter_basis_forget(M_gain=M_gain, ..., score_variant="combined")
   (same call as in current run_trajectory, lines 455–486)

3. Build search basis B for gradient ascent on the HM combination:
     union = vstack(A_w, A_{w+1})
     Q_union = rowspace_basis(union)               # rank ≤ 64
     Q_sketch = rowspace_basis(B_top) if state else empty
     # Include sketch in basis so ascent can find directions in M_gain rowspace
     # that overlap A_{w+1} support.
     Q_full = orthonormalize_columns([Q_union, Q_sketch])  # union of both rowspaces
     B = orth_basis_against(Q_full, v1)             # ⊥ v1, ⊥ ||·|| = 1

4. Compute denominators for HM-share normalizers, fixed at the candidate
   pool's max (matches existing online conventions). See
   hmean_combinations_optimizer_diagnostic.candidate_denoms — re-use that.

5. Build seed list for gradient ascent:
     seeds = [V_default[:, 1]]                     # warm start from combined v_carry
     seeds += build_candidates(...).values()       # opt2_outside, mgain_deflated_svd,
                                                   # block_complement, prev_opt2
     seeds += diag.get("Vbasis_final") columns     # optimizer's final reduced basis
     # Do NOT include q2_vs_q1oracle or q2_raw_projected — those are oracle.

6. Run gradient ascent with `optimize_combination_in_basis` from
   hmean_combinations_optimizer_diagnostic.py:
     best = optimize_combination_in_basis(
         policy, A_w, A_{w+1}, A_sketch_prior, denoms, weights,
         B, seeds, rng, maxit=80, tol=1e-9, random_starts=8,
     )
     v2 = best["vec"]

7. Form V_selected:
     V = column_stack([v1, v2])
     V_selected = orthonormalize_columns(V)
   Optionally: apply rank2_svd_frame(v1, v2, M_gain) to reframe in
   descending singular order, matching the existing online policies (lines
   526–530 in run_trajectory).

8. Update state via make_state (unchanged):
     compute score_selected, H_selected for V_selected
     state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)

9. Update old_row_memory via probe.select_old_row_memory (unchanged).
```

### Components already available — reuse, don't re-implement

- `cex_restricted_space_probe.entropy_iter_basis_forget` — combined-score
  optimizer that gives `v1`.
- `cex_restricted_space_probe.row_norm_seed` — initial frame.
- `cex_restricted_space_probe.score_full_vector_details_forget` — for
  computing `score_selected` and `H_selected` to pass to make_state.
- `cex_restricted_space_probe.select_old_row_memory` — old-row memory update.
- `second_slot_tail_bias_diagnostic.make_state` — sketch update via
  `left_projected_operator_svd_factors`.
- `hmean_combinations_optimizer_diagnostic.combination_value_grad` —
  per-policy value/grad function.
- `hmean_combinations_optimizer_diagnostic.optimize_combination_in_basis` —
  Riemannian gradient ascent driver. Re-uses
  `future_hmean_optimizer_diagnostic.optimize_future_hmean_in_basis` with the
  value/grad function swapped via globals patching.
- `hmean_combinations_optimizer_diagnostic.candidate_denoms` — fixed share
  denominators.
- `half_window_sliding_hmean_experiment.build_candidates` — for seeds (NOT
  for re-ranking).
- `half_window_sliding_hmean_experiment.rank2_svd_frame` — optional reframe.

### HM combinations to implement (initial set)

```
triplet:  HM(sk_share, gain1_share, gain2_share) * relH1
nested:   HM(c1_share, gain2_share) * relH1
            with c1_share = ||[sk; A_w] v||² / max-across-pool
weighted: weighted-HM([sk_share, gain1_share, gain2_share],
                      weights=(rows_seen_in_sketch, |A_w|, |A_{w+1}|)) * relH1
```

Skip `pairwise` — confirmed structurally similar to combined-score (both HM
arguments contain sketch).

The `relH1` factor uses entropy on `A_w v` only; preserve that to match the
existing implementation. Both `_no_phi` variants drop `relH1` and are worth
testing separately for ablation but are secondary.

### Critical: do NOT include oracle in the seed list or the search basis

The existing `future_hmean_nested_online` policy is effectively an oracle
because `q2_vs_q1oracle ∈ ONLINE_POOL`. The whole point of this
implementation is to test whether the HM combinations can produce the right
direction *without* oracle. Specifically, exclude these candidates from the
seed list:

- `q2_vs_q1oracle`
- `q2_raw_projected`
- `opt2_outside` (uses `Q_oracle`)

Allowed seeds: `opt2`, `mgain_deflated_svd`, `block_complement`, `prev_opt2`,
`half2_complement` (the SVD complement of A_{w+1} — this is fair because
A_{w+1} is part of the search basis already), V_default[:, 1], random
restarts.

## How to evaluate

Run on the eight tail-dominant matrices used in
`future_hmean_optimizer_diagnostic` (`mixed-tail-sharp`, `mixed-tail-balanced`,
`mixed-tail-soft`, `diffuse-diffuse`, `static-cex`, `etf-basket-basis`,
`residual-spiky-shocks`, `risk-residual-panel`) at `n=1024`, `half_win=32`,
`rank=2`, full block range (do not cap to 8 blocks — we need enough blocks
for the noisy power iteration to converge).

For each matrix and each policy in `{triplet, nested, weighted}`, record per
block:

- `block`, `rows_seen`
- `v2_align_exact = |⟨v2, V_exact[:,1]⟩|²`
- `state_V0_align_exact = |⟨state["V"][:,0], V_exact[:,0]⟩|²`
- `state_V1_align_exact = |⟨state["V"][:,1], V_exact[:,1]⟩|²`
- `mean_relerr_sval` (compare state["s"] to true singular values)
- `final_tail_mass`
- `gradient_ascent_iterations`, `gradient_ascent_grad_norm` at termination
- `policy_score_at_v2` (the gradient-ascent terminal value)

Compare against three baselines:

1. `combined`: existing combined-score-only run (`run_trajectory(..., policy="combined")`).
2. `future_hmean_nested_online` (rerank, oracle-leaking): existing
   implementation. This is the upper-bound oracle reference.
3. `future_hmean_<policy>_online` (rerank, no oracle): existing rerank with
   oracle candidates removed from the pool, to isolate "rerank vs gradient
   ascent" with the same candidate building blocks.

The key plot is `state_V1_align_exact` vs `block` — does it look like noisy
power iteration converging? Expected:

- etf-basket-basis: converges fast (one block has alignment ≈ 0.25 already).
- mixed-tail-sharp / mixed-tail-balanced / mixed-tail-soft: slow ascent over
  many blocks. If the iteration is right, `state_V1_align_exact` should rise
  monotonically (noisy) and asymptote.
- The same plot under `combined` should stay near 0.045 (no update is fed in).

If the gradient-ascent variant approaches `nested_online` (oracle) within a
significant fraction of blocks, the framing in this spec is empirically
confirmed: the HM combinations were the right *objectives* all along, and
the rerank-only implementation was the wrong wrapper.

## File list

Create:

- `half_window_sliding_hmean_gradient.py` — the new runner. Mirror
  `run_trajectory` structure but replace the rerank step with gradient
  ascent (steps 3–7 above).

Save outputs to:

- `summary/hmean_gradient_ascent_streaming/<matrix>_<policy>.csv`
- `summary/hmean_gradient_ascent_streaming/<matrix>_<policy>.json`
- `summary/hmean_gradient_ascent_streaming/synthesis.txt`

Do not modify:

- `cex_restricted_space_probe.py`
- `second_slot_tail_bias_diagnostic.py`
- `half_window_sliding_hmean_experiment.py` (keep it as the rerank baseline)
- `hmean_combinations_optimizer_diagnostic.py` (re-use its functions
  directly; do not duplicate the gradient code)

## Sanity checks before running the full sweep

1. On block 1 of `mixed-tail-sharp`, with `state=None`, gradient ascent on
   nested HM should produce a v2 with nonzero `||A_{w+1} v2||²`. Print
   `gain1`, `gain2`, `relH1` at termination — gain2 should be O(0.1) not
   O(1e-5).
2. On block 2 (sketch present), v2's mass distribution should be: nonzero
   in S1 (state V), nonzero in S3 (A_w residual), nonzero in S4 (A_{w+1}
   residual). All three should be > 0.01.
3. The gradient norm at termination should be ≤ 1e-6 for at least 80% of
   blocks; if many blocks bail with `line_search_fail`, increase
   `random_starts` or `maxit`.
4. On `etf-basket-basis`, even just blocks 1–3 should produce
   `state_V1_align_exact > 0.5`. If that doesn't happen, the implementation
   has a bug.

## What this is NOT testing

This spec is about whether HM-combinations as *streamed gradient-ascent
objectives* recover slot 2. It is not testing:

- Whether the HM combinations are good first-slot objectives (we keep
  combined-score for v1 throughout).
- Whether HM combinations help when we don't carry state (always-fresh setup).
- Whether the right `relH1` weighting can be tuned to outperform oracle.
- Whether the inner combined-score optimizer should be replaced too.

Those are downstream questions. The first thing to know is whether the
streaming-gradient-ascent version even converges on slot 2 across the
tail-dominant matrices.
