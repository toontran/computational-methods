"""Full-suite runner for defsvd-varshrink: all 18 BENCHMARK_MATRICES + the 3
static-cex stress variants, at rank in {2, 8}, l in {1, rank}, lambda sweep.

Produces an honest cross-suite win/tie/loss tally vs isvd / fd / carryonly,
not just the 3-matrix discriminator slice.

    cd /home/ttran02/pj/computational-methods/test_matrices_fast
    python run_varshrink_full_benchmark.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import benchmark_defsvd as bd
from cex_restricted_space_probe import generate_matrix_input

# Full canonical suite + the 3 stress variants (failure cases listed first
# within the variants for emphasis; order does not affect per-matrix results).
STRESS = ["static-cex-noisy", "static-cex-gauss", "static-cex-exptail"]
MATRICES = STRESS + list(bd.BENCHMARK_MATRICES)

LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]
RANKS = [2, 8]
N = 128
WIN = 32
SEED = 0
PRESET = "fast"
TOL = 1e-6  # win/tie threshold on alignment


def run_one(A, V_exact, sigma1, mode, deflate_window, rank, l):
    np.random.seed(SEED)
    return bd.run_streaming(A, V_exact, sigma1, rank, WIN, mode, deflate_window, l=l)


def main():
    out = []

    def emit(s=""):
        print(s, flush=True)
        out.append(s)

    emit(f"# DefSVD-VarShrink FULL suite  n={N} win={WIN} seed={SEED} preset={PRESET}")
    emit(f"# ranks={RANKS}  lambdas={LAMBDAS}  matrices={len(MATRICES)}")
    emit("")

    # Pre-generate. Skip any matrix that fails to generate (record it).
    mats, skipped = {}, []
    for m in MATRICES:
        try:
            np.random.seed(SEED)
            A, V_exact, _, sigma1 = generate_matrix_input(m, n=N, preset=PRESET, seed=SEED)
            mats[m] = (A, V_exact, sigma1)
        except Exception as e:  # noqa: BLE001
            skipped.append((m, repr(e)))
    if skipped:
        emit("## SKIPPED matrices (generation failed)")
        for m, e in skipped:
            emit(f"  {m:26s} {e}")
        emit("")
    live = [m for m in MATRICES if m in mats]

    refs = [("isvd-ref", "iSVD", True), ("fd-ref", "FD", True),
            ("defsvd-carryonly", "DefSVD", False)]

    for rank in RANKS:
        emit(f"\n{'='*72}\n## RANK = {rank}\n{'='*72}")
        ref_res, vs_res = {}, {}
        for m in live:
            A, V_exact, sigma1 = mats[m]
            for label, mode, dw in refs:
                for l in (1, rank):
                    ref_res[(m, label, l)] = run_one(A, V_exact, sigma1, mode, dw, rank, l)
        for lam in LAMBDAS:
            os.environ["VARSHRINK_LAMBDA"] = repr(lam)
            for m in live:
                A, V_exact, sigma1 = mats[m]
                for l in (1, rank):
                    vs_res[(m, lam, l)] = run_one(
                        A, V_exact, sigma1, "DefSVD-VarShrink", False, rank, l)

        # Per-matrix tables.
        for l in (1, rank):
            emit(f"\n### Alignment table  rank={rank} l={l}")
            hdr = f"  {'matrix':26s} {'isvd':>9s} {'fd':>9s} {'carryonly':>9s}" + \
                  "".join(f"{'vs@'+repr(x):>10s}" for x in LAMBDAS) + f"{'best_vs':>9s}"
            emit(hdr)
            for m in live:
                isvd_a = ref_res[(m, "isvd-ref", l)][0]
                fd_a = ref_res[(m, "fd-ref", l)][0]
                co_a = ref_res[(m, "defsvd-carryonly", l)][0]
                vs = [vs_res[(m, lam, l)][0] for lam in LAMBDAS]
                best = max(vs)
                emit(f"  {m:26s} {isvd_a:9.4f} {fd_a:9.4f} {co_a:9.4f}" +
                     "".join(f"{x:10.4f}" for x in vs) + f"{best:9.4f}")

        # Cross-suite win/tie/loss of best-lambda varshrink vs each baseline.
        emit(f"\n### Cross-suite tally (best-lambda varshrink vs baseline), rank={rank}")
        for l in (1, rank):
            for base in ("isvd-ref", "fd-ref", "defsvd-carryonly"):
                win = tie = loss = 0
                deltas = []
                for m in live:
                    base_a = ref_res[(m, base, l)][0]
                    best = max(vs_res[(m, lam, l)][0] for lam in LAMBDAS)
                    d = best - base_a
                    deltas.append(d)
                    if d > TOL:
                        win += 1
                    elif d < -TOL:
                        loss += 1
                    else:
                        tie += 1
                emit(f"  l={l:>2}  vs {base:18s}  W/T/L = {win:2d}/{tie:2d}/{loss:2d}"
                     f"   mean Δalign = {np.mean(deltas):+.4f}  "
                     f"median Δ = {np.median(deltas):+.4f}")

        # Honest: also report fixed-lambda (not cherry-picked) at lam=1.
        emit(f"\n### Fixed lambda=1 (no per-matrix cherry-pick) vs isvd, rank={rank}")
        for l in (1, rank):
            win = tie = loss = 0
            deltas = []
            for m in live:
                isvd_a = ref_res[(m, "isvd-ref", l)][0]
                d = vs_res[(m, 1.0, l)][0] - isvd_a
                deltas.append(d)
                if d > TOL:
                    win += 1
                elif d < -TOL:
                    loss += 1
                else:
                    tie += 1
            emit(f"  l={l:>2}  lam=1 vs isvd  W/T/L = {win:2d}/{tie:2d}/{loss:2d}"
                 f"   mean Δalign = {np.mean(deltas):+.4f}")

    Path(HERE / "varshrink_full_run_output.txt").write_text("\n".join(out) + "\n")
    emit("\n# wrote varshrink_full_run_output.txt")


if __name__ == "__main__":
    main()
