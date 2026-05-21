# Running `future_hmean_online` efficiently — handoff guide

Working dir: `test_matrices_fast/`. Python deps: `numpy`, `scipy`. No build step.

## 1. Quick start — run on a built-in matrix

The 4-way comparison harness is `half_window_sliding_hmean_experiment.py`. Defaults are already tuned for speed (`--num-restarts 3 --patience 5 --patience-rel-tol 1e-5`), so a plain invocation runs efficiently:

```bash
cd test_matrices_fast
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python half_window_sliding_hmean_experiment.py \
  --matrix mixed-tail-sharp \
  --policies combined hybrid isvd future_hmean_online \
  --half-win 32 \
  --json-out summary/mine/<name>_win64.json \
  --csv-out  summary/mine/<name>_win64.csv \
  --text-out summary/mine/<name>_win64.txt
```

`<name>_win64.txt` has the 8 sliding-mode summary lines (4 policies × {split_probe, sliding}). `<name>_win64.csv` has per-block trajectory metrics. The 17 built-in matrices and the per-policy regime expectations are in `summary/bench_matrix_sweep/synthesis.txt`. Cap BLAS threads when running multiple matrices in parallel — without the cap, OpenBLAS oversubscribes and inflates wall-clock 2-3×.

## 2. Add a new matrix family

Two edits in `cex_restricted_space_probe.py`:

**a)** Implement a generator that returns `(A, V_exact, svec, sigma1)`:

```python
def generate_my_matrix_input(n=1024, preset="fast", seed=0):
    rng = np.random.default_rng(seed)
    # ... build A of shape (n, d) ...
    # V_exact = ground-truth right singular vectors of A, shape (d, k)
    # svec    = ground-truth singular values, shape (k,) or full
    # sigma1  = top singular value (float)
    return A, V_exact, svec, sigma1
```

`V_exact[:, :rank]` is the ground truth the harness scores against (`exact_cos1`, `exact_cos2`, `final_tail_mass`, `mean_relerr_sval`). If your matrix has no analytic ground truth, take a full `np.linalg.svd(A)` once at generation time and use those columns.

**b)** Register it in `generate_matrix_input` (`cex_restricted_space_probe.py:3504`):

```python
elif matrix == "my-matrix":
    A, V_exact, svec, sigma1 = generate_my_matrix_input(n=n, preset=preset, seed=seed)
```

Then run `--matrix my-matrix`. The harness's `--matrix` flag is a free-form string, no choices list to update.

## 3. What the output means (sliding mode is the one that matters)

For each policy:

| field | meaning |
|---|---|
| `mean_exact_cos1`, `mean_exact_cos2` | mean over blocks of cosine between `V_selected` and `V_exact[:, :rank]` |
| `final_exact_cos1`, `final_exact_cos2` | same, last block only |
| `final_tail_mass` | `1 − frac. of V_selected energy inside V_exact[:, :rank]` (lower = better) |
| `mean_relerr_sval` | `abs(top_sval_est − sigma1) / sigma1` at final block |
| `elapsed`, `sec_per_step` | wall-clock for that policy's run |

Match `future_hmean_online` against `isvd` head-to-head:

- If iSVD already ties or wins on `mean_exact_cos1` → well-conditioned regime, use iSVD (~9× faster).
- If `future_hmean_online` wins on `final_tail_mass` and `mean_exact_cos2` while iSVD has cos2 ≪ 1 → tail-dominant regime, online is right.
- If `future_hmean_online` does *worse* than iSVD on both quality metrics → spiky-residual regime, the score misreads the spike as noise; use iSVD.

## 4. Aggregating multiple matrices

Drop your per-matrix JSONs in `summary/bench_matrix_sweep/<name>_win64.json` and run `python summary/bench_matrix_sweep/aggregate.py` — it produces the per-metric table, win counts, and W/T/L head-to-head used in `synthesis.txt`. `run_bench_sweep_v2.sh` is the driver script that runs the full 17 matrices in 4-way parallel batches; copy and edit the matrix list to swap in new ones.

## 5. Knobs worth knowing

Defaults already give the speedup; only touch these if you want to explore:

| flag | default | what it does |
|---|---|---|
| `--num-restarts` | 3 | restart seeds for the cex ascent. With the explicit `V_init` + `prev_basis` seeds, 3 fills the budget so `make_basic_restart_seeds`'s SVD is skipped on every expansion after the first per slot. |
| `--patience` | 5 | break the ascent after K consecutive iters of < `--patience-rel-tol` relative score change. Set 0 to disable. |
| `--patience-rel-tol` | 1e-5 | relative-change threshold for the patience counter. |
| `--maxit` / `--post-expansion-maxit` | 120 / 80 | hard caps on ascent iters; patience usually fires first. |
| `--half-win` | 16 (CLI) / 32 (bench) | block size; pair window is `2*half_win`. The bench at 32 is what `synthesis.txt` reports. |
| `--n` | 1024 | matrix dimension. |
| `--rank` | 2 | active rank tracked per block. |

## 6. Things to know

- The online policy currently lives only in `half_window_sliding_hmean_experiment.py` and `compare_future_hmean_window.py`. `cex_restricted_space_probe.py`'s standalone `--mode` CLI does not have an online option — it's the engine library, not the online driver.
- `compare_future_hmean_window.py` still has `--num-restarts 8` default and no `--patience` flag (the seed improvements still apply because they live in the engine, but patience doesn't). Use the `half_window_*` harness if you can.
- Two known underperformer regimes for online: `residual-spiky-shocks` and `risk-residual-panel`. If your matrix's signal is itself a sparse row-spike, expect online to lose to iSVD.
- All quality metrics depend on `V_exact` from the generator — don't skip step 2(a)'s ground-truth wiring or the metrics will be NaN/garbage.
