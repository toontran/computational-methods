"""FAM-03 E0 S-2 wrapper.

Adds the E0 sum-of-trace objective to the existing frame oracle-vs-winner
screen without changing shared FAM-01-DIAG code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_frame_oracle_gap as frame_gap


ABLATION = "fam03_e0_sum_trace"


_orig_frame_score = frame_gap.frame_score
_orig_frame_value_grad = frame_gap.frame_value_grad


def _e0_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V):
    V = np.asarray(V, dtype=np.float64)
    val = 0.0
    grad = np.zeros_like(V)
    terms = (
        (A_sk, sk_F2),
        (A_cur, cur_F2),
        (A_fut, fut_F2),
    )
    for A, F2 in terms:
        if A is None or np.asarray(A).size == 0 or F2 <= 0.0:
            continue
        Y = A @ V
        val += float(np.sum(Y * Y) / F2)
        grad += (2.0 / F2) * (A.T @ Y)
    return val, grad


def _patched_frame_score(Z, *, A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2,
                         ablation="hm"):
    if ablation != ABLATION:
        return _orig_frame_score(
            Z,
            A_sk=A_sk,
            A_cur=A_cur,
            A_fut=A_fut,
            sk_F2=sk_F2,
            cur_F2=cur_F2,
            fut_F2=fut_F2,
            ablation=ablation,
        )
    val, _ = _e0_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, Z)
    return val


def _patched_frame_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V,
                              ablation="hm"):
    if ablation != ABLATION:
        return _orig_frame_value_grad(
            A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V,
            ablation=ablation,
        )
    return _e0_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V)


def main():
    if "--gradient-check" in sys.argv:
        raise SystemExit(
            "E0 T1 gradient-check is not wired in this wrapper yet. "
            "Use this wrapper for S-2 only."
        )
    frame_gap.frame_score = _patched_frame_score
    frame_gap.frame_value_grad = _patched_frame_value_grad
    if "--quick" in sys.argv and "--matrices" not in sys.argv:
        sys.argv.extend(["--matrices", "diffuse-diffuse"])
    if "--ablations" not in sys.argv:
        sys.argv.extend(["--ablations", ABLATION])
    frame_gap.main()


if __name__ == "__main__":
    main()
