"""Runner for the defsvd-varshrink (DefSVD-VarShrink) sweep.

Drives run_streaming() directly so we can hit both l=1 and l=rank, and sweep
the shrinkage knob lambda in {0, 0.5, 1, 2, 5} via the VARSHRINK_LAMBDA env var.

Usage:
    cd /home/ttran02/pj/computational-methods/test_matrices_fast
    python run_varshrink_benchmark.py
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


# Failure cases FIRST, then hadamard, then financials.
MATRICES = [
    "static-cex-noisy",
    "static-cex-gauss",
    "static-cex-exptail",
    "static-cex",
    "mixed-tail-sharp",
    "crowded-strategy",
]

LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]

# Benchmark defaults (match argparse defaults in benchmark_defsvd.main).
N = 128
WIN = 32
RANK = 2
SEED = 0
PRESET = "fast"


def run_one(A, V_exact, sigma1, mode, deflate_window, l):
    np.random.seed(SEED)
    return bd.run_streaming(A, V_exact, sigma1, RANK, WIN, mode, deflate_window, l=l)


def main():
    out_lines = []

    def emit(s=""):
        print(s, flush=True)
        out_lines.append(s)

    emit(f"# DefSVD-VarShrink sweep  n={N} win={WIN} rank={RANK} seed={SEED} preset={PRESET}")
    emit(f"# lambdas={LAMBDAS}")
    emit("")

    # Pre-generate matrices.
    mats = {}
    for matrix in MATRICES:
        np.random.seed(SEED)
        A, V_exact, _, sigma1 = generate_matrix_input(matrix, n=N, preset=PRESET, seed=SEED)
        mats[matrix] = (A, V_exact, sigma1)

    # ---- Sanity: lambda=0 == isvd to machine precision ----
    emit("## Sanity: VARSHRINK_LAMBDA=0 vs isvd-ref")
    os.environ["VARSHRINK_LAMBDA"] = "0"
    max_align_diff = 0.0
    max_relerr_diff = 0.0
    for matrix in MATRICES:
        A, V_exact, sigma1 = mats[matrix]
        for l in (1, RANK):
            ai, ri, _ = run_one(A, V_exact, sigma1, "iSVD", True, l)
            av, rv, _ = run_one(A, V_exact, sigma1, "DefSVD-VarShrink", False, l)
            da = abs(ai - av)
            dr = abs(ri - rv)
            max_align_diff = max(max_align_diff, da)
            max_relerr_diff = max(max_relerr_diff, dr)
            emit(f"  {matrix:22s} l={l}  align isvd={ai:.10f} vs={av:.10f}  |d|={da:.2e}   "
                 f"relerr isvd={ri:.6e} vs={rv:.6e} |d|={dr:.2e}")
    emit(f"  MAX |align diff| = {max_align_diff:.3e}   MAX |relerr diff| = {max_relerr_diff:.3e}")
    emit("")

    # ---- Reference methods (lambda-independent) ----
    refs = [
        ("isvd-ref", "iSVD", True),
        ("fd-ref", "FD", True),
        ("defsvd-carryonly", "DefSVD", False),
    ]
    ref_results = {}  # (matrix, label, l) -> (align, relerr, elapsed)
    for matrix in MATRICES:
        A, V_exact, sigma1 = mats[matrix]
        for label, mode, dw in refs:
            for l in (1, RANK):
                ref_results[(matrix, label, l)] = run_one(A, V_exact, sigma1, mode, dw, l)

    # ---- VarShrink sweep ----
    vs_results = {}  # (matrix, lam, l) -> (align, relerr, elapsed)
    for lam in LAMBDAS:
        os.environ["VARSHRINK_LAMBDA"] = repr(lam)
        for matrix in MATRICES:
            A, V_exact, sigma1 = mats[matrix]
            for l in (1, RANK):
                vs_results[(matrix, lam, l)] = run_one(
                    A, V_exact, sigma1, "DefSVD-VarShrink", False, l)

    # ---- Timing: average elapsed per method (rank, full streaming) ----
    emit("## Runtime (mean over matrices, l=rank, seconds)")
    timing = {}
    for label, mode, dw in refs:
        ts = []
        for matrix in MATRICES:
            A, V_exact, sigma1 = mats[matrix]
            t0 = time.time()
            for _ in range(5):
                run_one(A, V_exact, sigma1, mode, dw, RANK)
            ts.append((time.time() - t0) / 5.0)
        timing[label] = float(np.mean(ts))
    os.environ["VARSHRINK_LAMBDA"] = "1.0"
    ts = []
    for matrix in MATRICES:
        A, V_exact, sigma1 = mats[matrix]
        t0 = time.time()
        for _ in range(5):
            run_one(A, V_exact, sigma1, "DefSVD-VarShrink", False, RANK)
        ts.append((time.time() - t0) / 5.0)
    timing["defsvd-varshrink(lam=1,M=16)"] = float(np.mean(ts))
    for k, v in timing.items():
        emit(f"  {k:34s} {v*1000:.3f} ms")
    emit("")

    # ---- Tables per matrix ----
    for l in (1, RANK):
        emit(f"## Alignment tables  (l={l})")
        emit("")
        for matrix in MATRICES:
            emit(f"### {matrix}  (l={l})")
            emit(f"  {'method':28s} {'align':>10s} {'relerr_sval':>12s}")
            for label in ("isvd-ref", "fd-ref", "defsvd-carryonly"):
                a, rr, _ = ref_results[(matrix, label, l)]
                emit(f"  {label:28s} {a:10.6f} {rr:12.4e}")
            for lam in LAMBDAS:
                a, rr, _ = vs_results[(matrix, lam, l)]
                emit(f"  {'varshrink lam=' + repr(lam):28s} {a:10.6f} {rr:12.4e}")
            emit("")

    # ---- Discriminator summary ----
    emit("## Discriminator summary (noisy / gauss / exptail): isvd vs best-lambda varshrink")
    discrim = ["static-cex-noisy", "static-cex-gauss", "static-cex-exptail"]
    for l in (1, RANK):
        emit(f"  l={l}:")
        for matrix in discrim:
            isvd_a = ref_results[(matrix, "isvd-ref", l)][0]
            best_lam, best_a = None, -1.0
            for lam in LAMBDAS:
                a = vs_results[(matrix, lam, l)][0]
                if a > best_a:
                    best_a, best_lam = a, lam
            verdict = "VARSHRINK WINS" if best_a > isvd_a + 1e-9 else "isvd wins/tie"
            emit(f"    {matrix:22s} isvd={isvd_a:.6f}  best_varshrink={best_a:.6f} "
                 f"(lam={best_lam})  -> {verdict}")
    emit("")

    # ---- Large-lambda -> equal-weight check ----
    emit("## Large-lambda -> equal-weight behavior (relerr_sval growth as lam grows, l=rank)")
    for matrix in MATRICES:
        relerrs = [vs_results[(matrix, lam, RANK)][1] for lam in LAMBDAS]
        emit(f"  {matrix:22s} relerr@lam[{LAMBDAS}] = "
             + "[" + ", ".join(f"{x:.3e}" for x in relerrs) + "]")
    emit("")

    Path(HERE / "varshrink_run_output.txt").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
