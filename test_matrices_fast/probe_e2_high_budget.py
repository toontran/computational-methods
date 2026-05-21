"""Probe F (AB-03 phase 1 closure follow-up).

Question: my Probe A reported Δ_S6_E2(slot-2 winner − oracle_v2_proj) =
+0.041 / +0.047 on diffuse-diffuse / residual-spiky-shocks at b31 using
the standard bench rank-1 sphere optimizer (24 random starts, maxit=120).
Could that gap be an artifact of (a) too few restarts, (b) a non-oracle
slot-1 anchor making B_def less useful, or (c) sequential greed missing
a joint-better solution?

This script re-runs S6 and S6_E2 slot-2 optimization with HIGH budget
(200 random starts, maxit=2000, tol=1e-12) under three slot-1 anchors:
  (1) default — slot-1 = high-budget S6 / S6_E2 winner over B_union
  (2) oracle  — slot-1 = oracle_v1_proj (forces deflation to respect
                 the oracle plane; anchors the question of "with the
                 RIGHT slot-1, does slot-2 land on oracle_v2?")
  (3) sketch  — slot-1 = V_state[:,0]
For each anchor, we report:
  score_S6_E2(slot-2 winner)
  score_S6_E2(oracle_v2_proj)
  Δ = winner − oracle
  cos²(slot-2 winner, oracle_v2_proj)

Reads: r_sk_g_score (optimizer + value/grad), per_block_constants,
       stream_to_block.
Writes: summary/score_family_aggregator_ablation/highbudget_<matrix>_b31.txt
        summary/score_family_aggregator_ablation/highbudget_summary.txt
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment  # noqa: F401
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import (
    _state_V,
    build_e2_data,
    optimize_r_sk_g_in_basis,
    r_sk_g_value_grad,
)
from probe_e2_landscape import make_default_args


def _proj_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    n = float(np.linalg.norm(p))
    return None if n <= 1e-30 else p / n


def _score(v, *, A_sk, A_c, A_f, c_sk, V_state, cur_F2, fut_F2, sk_F2_low,
           e2_data, variant):
    out = r_sk_g_value_grad(A_sk, A_c, A_f, c_sk, v,
                            variant=variant, V_state=V_state,
                            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
                            e2_data=e2_data)
    return float(out[0])


def run_for_matrix(matrix, *, block=31, random_starts=200, maxit=2000,
                   tol=1e-12, n_dense_restarts=64):
    args = make_default_args(matrix, block=block)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block, {block})
    snap = snapshots[block]
    consts = per_block_constants(A, block, args.half_win)

    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch = np.asarray(snap["A_sketch"], dtype=np.float64)
    A_sk_for = A_sketch if A_sketch.size else None
    state = snap["state"]
    V_state = _state_V(state)
    V_default = snap["V_default"]

    c_sk = float(consts["c_sk"])
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(A_sketch ** 2)) if A_sk_for is not None else 0.0

    e2_data = build_e2_data(A_cur, A_fut, A_sketch, state, int(args.rank))

    if A_sk_for is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    oracle_v1_proj = _proj_unit(V_exact[:, 0], B_union)
    oracle_v2_proj = _proj_unit(V_exact[:, 1], B_union)
    sketch_v1 = (V_state[:, 0] / max(np.linalg.norm(V_state[:, 0]), 1e-30)
                 if V_state is not None and V_state.shape[1] >= 1 else None)

    # Big warm-start pool: oracles + sketch + V_default + many random in B_union.
    rng = np.random.default_rng(args.seed + 9_999_999 + block)
    starts_base = []
    for v in (oracle_v1_proj, oracle_v2_proj, sketch_v1,
              V_default[:, 0], V_default[:, 1]):
        if v is not None:
            starts_base.append(np.asarray(v, dtype=np.float64))
    # Dense random in B_union (rejection-free) — these will be normalized
    # and re-projected by the optimizer.
    for _ in range(n_dense_restarts):
        z = B_union @ rng.standard_normal(B_union.shape[1])
        n = float(np.linalg.norm(z))
        if n > 1e-30:
            starts_base.append(z / n)

    def hi_opt(variant, basis, seed_off, extra_starts=None, **kwargs):
        starts = list(starts_base) + (extra_starts or [])
        best = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sk_for, c_sk,
            basis, starts,
            np.random.default_rng(args.seed + seed_off + block),
            maxit, tol, random_starts,
            variant=variant, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
            **kwargs,
        )
        if best is None:
            return None
        v = np.asarray(best["vec"], dtype=np.float64)
        return v / max(np.linalg.norm(v), 1e-30)

    out = {"matrix": matrix, "block": block, "rows": []}

    def score_e2(v):
        return _score(v, A_sk=A_sk_for, A_c=A_cur, A_f=A_fut, c_sk=c_sk,
                      V_state=V_state, cur_F2=cur_F2, fut_F2=fut_F2,
                      sk_F2_low=sk_F2_low, e2_data=e2_data, variant="S6_E2")

    def score_s6(v):
        return _score(v, A_sk=A_sk_for, A_c=A_cur, A_f=A_fut, c_sk=c_sk,
                      V_state=V_state, cur_F2=cur_F2, fut_F2=fut_F2,
                      sk_F2_low=sk_F2_low, e2_data=e2_data, variant="S6")

    o2_score_e2 = score_e2(oracle_v2_proj) if oracle_v2_proj is not None else float("nan")
    o2_score_s6 = score_s6(oracle_v2_proj) if oracle_v2_proj is not None else float("nan")

    # ---- Anchor 1: default (high-budget slot-1 winner over B_union) ----
    for variant_label, variant, score_fn in (("S6", "S6", score_s6),
                                              ("S6_E2", "S6_E2", score_e2)):
        kw = {"e2_data": e2_data} if variant == "S6_E2" else {}
        v1 = hi_opt(variant, B_union, 700_000, **kw)
        if v1 is None:
            continue
        v1_score = score_fn(v1)
        B_def = orth_basis_against(B_union, v1)
        v2 = hi_opt(variant, B_def, 800_000, **kw)
        if v2 is None:
            continue
        v2_score = score_fn(v2)
        cos2_o1 = float(np.dot(v1, oracle_v1_proj) ** 2) if oracle_v1_proj is not None else float("nan")
        cos2_o2 = float(np.dot(v2, oracle_v2_proj) ** 2) if oracle_v2_proj is not None else float("nan")
        out["rows"].append({
            "anchor": "default", "variant": variant_label,
            "v1_score": v1_score, "v2_score": v2_score,
            "oracle_v2_score": o2_score_e2 if variant == "S6_E2" else o2_score_s6,
            "delta_v2_minus_oracle": (v2_score - (o2_score_e2 if variant == "S6_E2" else o2_score_s6)),
            "cos2_v1_oracle1": cos2_o1, "cos2_v2_oracle2": cos2_o2,
        })

    # ---- Anchor 2: slot-1 = oracle_v1_proj (deflation respects oracle frame) ----
    if oracle_v1_proj is not None:
        B_def_oracle = orth_basis_against(B_union, oracle_v1_proj)
        for variant_label, variant, score_fn in (("S6", "S6", score_s6),
                                                  ("S6_E2", "S6_E2", score_e2)):
            kw = {"e2_data": e2_data} if variant == "S6_E2" else {}
            v2 = hi_opt(variant, B_def_oracle, 900_000, **kw)
            if v2 is None:
                continue
            v2_score = score_fn(v2)
            cos2_o2 = float(np.dot(v2, oracle_v2_proj) ** 2) if oracle_v2_proj is not None else float("nan")
            out["rows"].append({
                "anchor": "oracle_v1", "variant": variant_label,
                "v1_score": float("nan"), "v2_score": v2_score,
                "oracle_v2_score": o2_score_e2 if variant == "S6_E2" else o2_score_s6,
                "delta_v2_minus_oracle": (v2_score - (o2_score_e2 if variant == "S6_E2" else o2_score_s6)),
                "cos2_v1_oracle1": float("nan"), "cos2_v2_oracle2": cos2_o2,
            })

    # ---- Anchor 3: slot-1 = sketch_v1 ----
    if sketch_v1 is not None:
        B_def_sketch = orth_basis_against(B_union, sketch_v1)
        for variant_label, variant, score_fn in (("S6", "S6", score_s6),
                                                  ("S6_E2", "S6_E2", score_e2)):
            kw = {"e2_data": e2_data} if variant == "S6_E2" else {}
            v2 = hi_opt(variant, B_def_sketch, 1_000_000, **kw)
            if v2 is None:
                continue
            v2_score = score_fn(v2)
            cos2_o2 = float(np.dot(v2, oracle_v2_proj) ** 2) if oracle_v2_proj is not None else float("nan")
            out["rows"].append({
                "anchor": "sketch_v1", "variant": variant_label,
                "v1_score": float("nan"), "v2_score": v2_score,
                "oracle_v2_score": o2_score_e2 if variant == "S6_E2" else o2_score_s6,
                "delta_v2_minus_oracle": (v2_score - (o2_score_e2 if variant == "S6_E2" else o2_score_s6)),
                "cos2_v1_oracle1": float("nan"), "cos2_v2_oracle2": cos2_o2,
            })
    return out


def write_text(result, path):
    lines = [f"== matrix={result['matrix']} block={result['block']} (high budget) =="]
    lines.append(f"  {'anchor':<12} {'variant':<7} {'v1_score':>10} {'v2_score':>10} "
                 f"{'oracle_v2':>10} {'Δ(v2−o)':>10} {'cos²(v1,o1)':>11} {'cos²(v2,o2)':>11}")
    for r in result["rows"]:
        lines.append(
            f"  {r['anchor']:<12} {r['variant']:<7} "
            f"{r['v1_score']:>10.4f} {r['v2_score']:>10.4f} "
            f"{r['oracle_v2_score']:>10.4f} {r['delta_v2_minus_oracle']:>+10.4f} "
            f"{r['cos2_v1_oracle1']:>11.3f} {r['cos2_v2_oracle2']:>11.3f}"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=[
        "diffuse-diffuse", "residual-spiky-shocks", "mixed-tail-soft",
    ])
    ap.add_argument("--block", type=int, default=31)
    ap.add_argument("--random-starts", type=int, default=200)
    ap.add_argument("--maxit", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--out-dir", default="summary/score_family_aggregator_ablation")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary_lines = ["# AB-03 phase 1 high-budget probe — H1 sanity check",
                     f"# block={args.block} random_starts={args.random_starts} "
                     f"maxit={args.maxit} tol={args.tol}",
                     ""]
    summary_lines.append(
        f"{'matrix':<24} {'anchor':<12} {'variant':<7} "
        f"{'v2_score':>10} {'oracle_v2':>10} {'Δ(v2−o)':>10} {'cos²(v2,o2)':>11}"
    )
    for m in args.matrices:
        print(f"== {m} ==", flush=True)
        r = run_for_matrix(m, block=args.block,
                           random_starts=args.random_starts,
                           maxit=args.maxit, tol=args.tol)
        path = os.path.join(args.out_dir, f"highbudget_{m}_b{args.block}.txt")
        write_text(r, path)
        for row in r["rows"]:
            summary_lines.append(
                f"{m:<24} {row['anchor']:<12} {row['variant']:<7} "
                f"{row['v2_score']:>10.4f} {row['oracle_v2_score']:>10.4f} "
                f"{row['delta_v2_minus_oracle']:>+10.4f} "
                f"{row['cos2_v2_oracle2']:>11.3f}"
            )
        print(f"  wrote {path}", flush=True)

    summary_path = os.path.join(args.out_dir,
                                f"highbudget_summary_b{args.block}.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
