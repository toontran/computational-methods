# Frequent Directions in this codebase — primer for an outside reader

This document is self-contained. It assumes no prior exposure to the
project. All file paths are absolute.

Project root:
`/home/ttran02/pj/computational-methods/test_matrices_fast`

---

## 1. What Frequent Directions (FD) is

Frequent Directions is a deterministic streaming sketching algorithm
for matrices, due to Edo Liberty (KDD 2013, "Simple and Deterministic
Matrix Sketching"). For a tall data matrix `A ∈ R^(N × n)` arriving as
a stream of rows, FD maintains a small sketch `B ∈ R^(ℓ × n)` (with
`ℓ << N`) such that for every unit vector `v`:

```
0 ≤ ‖A v‖² − ‖B v‖² ≤ ‖A − A_k‖_F² / (ℓ − k)
```

i.e. `B` preserves squared row-projection energy in every direction up
to a bounded additive error, where `A_k` is the best rank-`k`
approximation of `A` and `k < ℓ`.

The canonical FD update (rank `ℓ`) for an arriving row `a`:

1. Append `a` as the last row of `B` (now `(ℓ+1) × n`).
2. Compute SVD: `B = U Σ V^T`.
3. Shrink: replace `Σ` with `sqrt(Σ² − σ_ℓ² · I)` (subtract the smallest
   squared singular value from each).
4. Truncate to top `ℓ−1` rows: `B ← Σ_{1:ℓ−1} V_{1:ℓ−1}^T`.

In practice the streaming variants differ in (a) whether they shrink,
(b) how rows are batched, (c) what the kept subspace is. The variant
in this codebase is a **batched, score-driven sketch** that generalizes
incremental SVD (iSVD); see §3.

---

## 2. Naming caveat — "FD" in this repo means two different things

Search results for "FD" in
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/`
hit BOTH meanings, and they are unrelated:

| Sense | Meaning | Where it appears |
| --- | --- | --- |
| **FD = finite difference** | gradient-correctness check on the sphere or Stiefel manifold (`(score(v + ε·u) − score(v))/ε` ≈ analytic gradient) | INFRA-02 (`stiefel_grad_check.py`); §2 of `diagnostic_toolkit.txt` |
| **FD = Frequent Directions** | the streaming sketch; in this repo realized as the **carry** `state["V"]`, the rank-`r` right-singular-vector basis maintained across blocks | this primer; the `isvd` policy in the streaming bench |

Be explicit when you write FD. In recent conversations the team has
used "FD direction" / "FD-kept direction" to mean the **carry**
(`V_state` in code), i.e. the second sense.

---

## 3. The streaming sketch this codebase actually maintains

### 3.1 The state object

At each block `t`, the streaming algorithm carries a state dict (built
by `make_state` in
`/home/ttran02/pj/computational-methods/test_matrices_fast/second_slot_tail_bias_diagnostic.py`,
function definition at line 16):

```python
state = {
    "V":         np.ndarray,    # shape (n, r) — orthonormal right-SV basis
    "s":         np.ndarray,    # shape (r,)   — singular values
    "s2":        np.ndarray,    # = s ** 2
    "H":         np.ndarray,    # entropy / score-aux per kept direction
    "score":     np.ndarray,    # score per kept direction
    "rows_seen": int,           # cumulative rows ingested
}
```

The "FD-kept directions" are the columns of `state["V"]` — usually
called `V_state[:, k]` or `sketch_v(k+1)` in the diagnostics.

### 3.2 The per-block update

Driver: `stream_to_block(args, A, V_exact, work_dtype, rank, target_block, all_blocks_to_report)`
at
`/home/ttran02/pj/computational-methods/test_matrices_fast/hmean_evidence_score.py:213`.

For block `t = 1..target_block` the loop does:

1. **Slice the data**:
   `A_cur = A[(t−1)·half_win : t·half_win]`,
   `A_fut = A[t·half_win : (t+1)·half_win]`.
   `half_win = 32` is the canonical bench setting (window 64).

2. **Build M_gain (the FD analogue of "B with the new row appended")**:
   ```
   B_top   = state["s"][:, None] * state["V"].T              # (r, n)
   M_gain  = vstack([B_top, A_cur])                          # (r + half_win, n)
   ```
   At `t = 1` there is no carry, so `M_gain = A_cur`.

3. **Pick a rank-r subspace** `V_score` of `R^n`. Two flavors:
   - **iSVD policy** — pure FD-style: `V_score = top-r right SVs of M_gain`.
     This is the closest variant to canonical Frequent Directions
     (without the singular-value shrinkage; see §3.3).
   - **Score-driven policies** (combined / hybrid / future_hmean / r_sk_g):
     `V_score` is the optimizer's best rank-r answer under a custom
     score `Score(v)` that combines `‖B v‖²`, `‖A_cur v‖²`,
     `‖A_fut v‖²` and an entropy term. See
     `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_design_overview.txt`
     for the score-design line.

4. **Project & SVD-truncate to update the carry**: build the new
   `(state.V, state.s)` from `V_score` and `M_gain` via
   `make_state` →
   `left_projected_operator_svd_factors` (or `projected_subspace_svd`)
   in
   `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py:2383`
   (and `:2372`). The resulting `state["V"]` is an orthonormal
   `(n × r)` basis that lives in the row span of `[B_top; A_cur]`,
   re-rotated to align with the chosen `V_score`.

5. **Old-row memory** (orthogonal to FD; used by some scores): a small
   set of past rows is retained for entropy diagnostics. See
   `select_old_row_memory` in
   `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py`.
   This is NOT part of the FD sketch itself.

### 3.3 Differences vs canonical Liberty-FD

| Aspect | Canonical FD | This codebase |
| --- | --- | --- |
| Append granularity | one row | a batch of `half_win = 32` rows |
| Singular-value shrinkage | yes (subtract `σ_ℓ²` from each) | **no** — closer to incremental SVD |
| Subspace selection | top `ℓ−1` right SVs of augmented `B` | iSVD policy: same. Score policies: optimizer's pick under custom score |
| Carries singular values? | yes (in `Σ`) | yes (`state["s"]`) |

The combination of (i) batched updates, (ii) no shrinkage, (iii) exact
top-`r` SVD truncation — that is the **incremental SVD (iSVD)** policy.
Without shrinkage, iSVD does not enjoy Liberty's
`‖A − A_k‖_F² / (ℓ − k)` bound. The score-family variants are a further
generalization where the kept subspace is the optimizer's argmax of a
non-SVD objective.

The "iSVD" baseline in the streaming bench
(`/home/ttran02/pj/computational-methods/test_matrices_fast/half_window_sliding_hmean_experiment.py`,
`--policies isvd`) is therefore the **canonical FD-like reference** in
this project.

---

## 4. Key files and entry points (full paths)

### Streaming algorithm
- Driver loop:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/hmean_evidence_score.py:213`
  (`stream_to_block`)
- Per-block constants (window F-norms):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/hmean_evidence_score.py:188`
  (`per_block_constants`)
- Carry-state builder:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/second_slot_tail_bias_diagnostic.py:16`
  (`make_state`)
- SVD primitives behind `make_state`:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py:2372`
  (`projected_subspace_svd`)
  `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py:2383`
  (`left_projected_operator_svd_factors`)
- Score-driven inner-loop optimizer:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py`
  (`entropy_iter_basis_forget`, plus the various `*_combined_*` /
  `*_future_hmean_*` value-and-grad routines)

### Streaming bench (ground truth)
- Driver:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/half_window_sliding_hmean_experiment.py`
- Policies of interest: `isvd` (canonical FD-like baseline),
  `combined`, `hybrid`, `future_hmean_online`,
  `future_hmean_r_sk_g` (S1..S6 score family).

### Diagnostics for inspecting FD-kept directions
- Direction-by-direction alignment probe (built 2026-05-04):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/direction_alignment_probe.py`
- Per-block snapshot table with `sketch_v?`, oracle, mgain candidates:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/r_sk_g_score.py`
- Component time-series (u_sk, u_g1, u_g2 of the carry over blocks):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/component_time_series_probe.py`
- Carry-trajectory metrics (s_top, spectral_concentration, drift):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/carry_trajectory_probe.py`
- Row-response entropy (relH1, eff_frac) of carry vs oracle:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/oracle_entropy_audit.py`
- Subsample stability of evidence:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/diag03_subsample_stability/probe.py`
- Subspace-alignment primitive:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/subspace_metrics.py`
  (`principal_angles(V_opt, V_oracle)`)
- Stiefel finite-difference gradient check (the *other* FD):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/stiefel_grad_check.py`

### Catalog of all diagnostics
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/diagnostic_toolkit.txt`

### Score-design design doc
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_design_overview.txt`

### Workflow / backlog
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_family_workflow.txt`

---

## 5. How to interpret the FD-kept directions

The carry `V_state` lives in ambient `R^n` (not `R^(rows_in_block)`).
Each column is a rank-1 right singular direction the algorithm has
chosen to remember. At rank `r = 2` (the canonical bench setting),
there are two FD-kept directions per block:

- `V_state[:, 0]` aka `sketch_v1` — leading carry direction.
- `V_state[:, 1]` aka `sketch_v2` — second carry direction.

By construction `V_state` lives in the row span of the *previous*
block's `M_gain = [B_top_{t−1}; A_cur_{t−1}]`, and after enough blocks,
it accumulates information from all historical rows.

### Useful reference directions to compare against (from §5 of `diagnostic_toolkit.txt`)

For every block, the per-block diagnostics also produce:

- `oracle_v?_exact` — population truth: `V_exact[:, k]` of the *full* matrix `A`.
- `oracle_v?_proj` — `V_exact[:, k]` projected into
  `B_union = rowspan([B; A_cur; A_fut])`, i.e. the **slotwise reachable
  target** in the visible window. For any unit `v ∈ B_union`:
  `cos²(v, oracle_exact) = ‖P_{B_union} V_exact[:, k]‖² · cos²(v, oracle_proj)`
  where the prefactor (the "oracle reachability") is a property of the
  matrix and block alone.
- `mgain_svd_v?` — top right SV of `M_gain = [B_top; A_cur]`. This is
  what the **iSVD policy** would have picked at this block — the
  canonical-FD-like rank-r choice.
- `combined_v1`, `combined_v2` — what the **combined-score
  optimizer** picked (the streaming algorithm's actual choice). The
  empirical fact verified at
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/...`
  / future_hmean_v2_subspace_decomposition.py is that `combined_v2`
  sits ~83% inside `span(V_state)` on most matrices — this is what the
  team calls the "outside-window direction = carried state"
  observation.

### Quick numeric reading of the carry

Run the new alignment probe (full path):

```
cd /home/ttran02/pj/computational-methods/test_matrices_fast
python direction_alignment_probe.py \
  --matrices static-cex mixed-tail-balanced diffuse-diffuse \
  --blocks 1 2 6 12 31 \
  --inputs sketch_v1 sketch_v2 combined_v1 combined_v2 \
           oracle_v1_exact oracle_v1_proj \
           oracle_v2_exact oracle_v2_proj
```

Outputs land in
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/infra_direction_alignment/`:

- per `(matrix, block)`: `<matrix>_b<block>_alignment.csv`,
  `<matrix>_b<block>_rowbias.csv`, `<matrix>_b<block>.txt` (pretty).
- aggregate: `synthesis_<matrix>.txt`, `synthesis_overview.txt`.

The pretty `.txt` shows the **oracle reachability per slot** in its
header and a wide cos²-table with the carry, optimizer pick, oracle
(both flavors), and per-block top SVs.

---

## 6. Recent empirical findings about FD-kept directions

Source:
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/infra_direction_alignment/synthesis_overview.txt`
(deterministic re-run with the seed-fix described in §7). Three
matrices probed at blocks 1, 2, 6, 12, 31, half_win = 32, rank = 2.

### 6.1 The carry never aligns with per-block top SVs

`cos²(sketch_v1, Acur_topSV_v1)` and `cos²(sketch_v1, Afut_topSV_v1)`
and `cos²(sketch_v1, visible_topSV_v1)` are essentially **0 on every
block of every matrix**. Same for `sketch_v2`. The carry occupies a
different subspace than any single block's leading right-SV. This is
a clean negative result: the FD sketch is NOT biased toward the
current block's principal direction.

### 6.2 Carry-vs-oracle alignment is regime-dependent

| matrix | sketch_v1 vs oracle_v1_proj at b31 | sketch_v2 vs oracle_v2_proj at b31 |
| --- | ---: | ---: |
| static-cex | 0.012 | **0.932** |
| mixed-tail-balanced | **0.921** | 0.032 |
| diffuse-diffuse | 0.101 | 0.060 |

- static-cex: carry slot-2 captures oracle slot-2 by b31. Slot-1 trapped.
- mixed-tail-balanced: carry slot-1 captures oracle slot-1; slot-2 misses.
- diffuse-diffuse: by b31 the carry has not yet locked onto either
  oracle slot in this 31-block horizon (oracle reach is also low at
  b31: slot-1 = 0.140, slot-2 = 0.480; the visible window simply does
  not contain enough of the population direction yet).

### 6.3 The optimizer's "outside" pick pins to the carry

State alignment of `combined_v2` (the streaming optimizer's slot-2
"outside-window" choice) at b31:

| matrix | cos²(combined_v2, V_state_frame) at b31 |
| --- | ---: |
| static-cex | **1.000** |
| mixed-tail-balanced | 0.941 |
| diffuse-diffuse | 0.953 |

By b31 the optimizer's outside pick lives ≥ 94% inside
`span(V_state)`. This is the **M2 mechanism** documented in §1bis of
`/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_design_overview.txt`
("rank-r carry pins slot-r to span(V_state)").

### 6.4 Oracle reachability is the gating quantity

`oracle_reach[k] = ‖P_{B_union} V_exact[:, k]‖²`. Trajectory at b31:

| matrix | slot-1 reach | slot-2 reach |
| --- | ---: | ---: |
| static-cex | 0.080 | 0.883 |
| mixed-tail-balanced | 0.750 | 0.077 |
| diffuse-diffuse | 0.140 | 0.480 |

The FD sketch can only point at the oracle to the extent the **visible
window** captures the oracle direction. An "exact"-scale alignment is
bounded above by reach × in-window alignment. Comparing
`cos²(v, oracle_v?_exact)` to `reach × cos²(v, oracle_v?_proj)`
verifies this on every line of the synthesis tables.

---

## 7. Reproducibility hazard and how to avoid it

`cex_restricted_space_probe.py` uses **unseeded** `np.random.standard_normal`
at lines 61, 87, 91, 95 (file:
`/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py`).
Without an explicit `np.random.seed(...)` call before
`stream_to_block`, two consecutive runs with identical args produce
**different** `V_state`. Observed swing on static-cex b31:
`cos²(sketch_v2, oracle_v2_proj)` ranged 0.000–0.932 across runs.

`direction_alignment_probe.py` calls
`np.random.seed(int(args.seed) + 7919 * int(block))` at the start of
`build_reference_pack`. Other probes built on `stream_to_block`
(`oracle_entropy_audit.py`, `component_time_series_probe.py`,
`plateau_width_probe.py`, …) **do not** seed and inherit the same
hazard. If you re-use any of them, add the same seed call before
relying on their numbers.

---

## 8. Glossary mapping conversational labels → code

| Said in conversation | Code identifier | Where defined |
| --- | --- | --- |
| FD direction / FD-kept | `state["V"][:, k]` / `V_state[:, k]` / `sketch_v(k+1)` | `make_state` at `/home/ttran02/pj/computational-methods/test_matrices_fast/second_slot_tail_bias_diagnostic.py:16` |
| iSVD direction (canonical FD-like) | `mgain_svd_v(k+1)` = top-r right SV of `M_gain` | snapshot in `r_sk_g_score.py` |
| outside (window) direction | `combined_v2` = `V_default[:, 1]` | snap field set by `stream_to_block` at `/home/ttran02/pj/computational-methods/test_matrices_fast/hmean_evidence_score.py:267` |
| oracle (population) | `oracle_v?_exact = V_exact[:, k]` | from `generate_matrix_input`, defined at `/home/ttran02/pj/computational-methods/test_matrices_fast/cex_restricted_space_probe.py:3538` |
| oracle (in-window achievable) | `oracle_v?_proj = P_{B_union} V_exact[:, k]` (renormalized) | per-block helper in many probes; canonical implementation at `/home/ttran02/pj/computational-methods/test_matrices_fast/row_cheat_baseline.py:146` (`oracle_frame_proj`) |
| FD (gradient check) | finite-difference score validation | `/home/ttran02/pj/computational-methods/test_matrices_fast/stiefel_grad_check.py` (rank-r) and `r_sk_g_score.py --gradient-check` (rank-1) |

---

## 9. Where to read next

- Catalog of every diagnostic with input/output schema:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/diagnostic_toolkit.txt`
- Score-design rationale (why the policies that drive the carry differ
  from canonical FD/iSVD):
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_design_overview.txt`
- Backlog and DAG of in-flight workflow items:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/overview/score_family_workflow.txt`
- Worked-example output of the new alignment probe across the three
  reference matrices:
  `/home/ttran02/pj/computational-methods/test_matrices_fast/summary/infra_direction_alignment/synthesis_overview.txt`
