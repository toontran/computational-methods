"""Mass-shift-gated streaming policy experiment.

At each block t, the gated policy picks slot-1 from
    {combined, iSVD}           (2-way variant)
    {combined, iSVD, oracle}   (3-way variant)
based on mass_shift = |‖A_cur·v_cur‖² − ‖A_cur·v_fut‖²| / max(...).

Slot-2 (when rank=2) is always taken from the combined optimizer.
make_state is applied to the assembled V_chosen to advance the carry.

We compare:
  pure_combined        — bench default
  pure_isvd            — slot-1 = top right SV of A_cur every block
  pure_oracle          — slot-1 = P_[sketch;A_cur] V_exact[:,0] (upper bound)
  gated_2way_τ         — pick combined if mass_shift < τ, else iSVD
  gated_3way_τ1_τ2     — combined < τ1; iSVD in [τ1,τ2); oracle ≥ τ2
"""

import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import make_state

DEFAULT_MATRICES = ["diffuse-diffuse", "mixed-tail-sharp", "mixed-tail-soft", "static-cex"]
DEFAULT_SEEDS = [0, 1, 2]
N_BLOCKS = 16
HALF_WIN = 32
N = 1024
RANK = 2
PRESET = "fast"


def make_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        matrix="placeholder",
        half_win=HALF_WIN, n=N, rank=RANK, preset=PRESET,
        seed=seed, shuffle_rows=True, row_shuffle_seed=seed,
        old_memory_size=32, dtype="float32",
        q0=8, qmax=48, krylov_depth=2, residual_tol=0.01,
        expansion_maxit=8, num_restarts=3, maxit=120, tol=1e-8,
        post_expansion_maxit=80, patience=5, patience_rel_tol=1e-5,
        r_sig=2, alpha_sig=0.003, alpha_tail=0.0145,
        tail_scale=0.99, sigma1=0.991, v_type="rand",
    )


def normed(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-30 else v


def rowspace_basis(M):
    if M is None or M.size == 0:
        return None
    _, s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    tol = max(M.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0.0)
    keep = s > tol
    return np.ascontiguousarray(Vt[keep].T) if keep.any() else None


def project_unit(target, B):
    if B is None or B.size == 0:
        return None
    p = B @ (B.T @ target)
    n = float(np.linalg.norm(p))
    return p / n if n > 1e-30 else None


def run_combined_optimizer(M_gain, A_block, state_prev, rank, seed_offset):
    work_dtype = np.float32
    V_init = probe.row_norm_seed(A_block, rank)
    V, _, _, _, _ = probe.entropy_iter_basis_forget(
        M_gain=np.asarray(M_gain, dtype=work_dtype),
        active_r=rank, rows_ref=N,
        V_init=np.asarray(V_init, dtype=work_dtype),
        q0=8, qmax=48, krylov_depth=2,
        residual_tol=0.01, expansion_maxit=8,
        num_restarts=3, maxit=120, tol=1e-8,
        rng=np.random.default_rng(seed_offset),
        verbose=False, state_prev=state_prev,
        A_block=np.asarray(A_block, dtype=work_dtype),
        rows_total=int(state_prev["rows_seen"] + A_block.shape[0]) if state_prev is not None else int(A_block.shape[0]),
        reduced_optimizer="cex", basis_selection="greedy",
        work_dtype=work_dtype, expansion_direction="residual",
        reuse_line_search_grad=True, expansion_warm_start=True,
        post_expansion_maxit=80,
        score_variant="combined", old_row_memory=None,
        combined_rank=None, patience=5, patience_rel_tol=1e-5,
    )
    return np.ascontiguousarray(np.asarray(V[:, :rank], dtype=np.float64))


def stream_with_policy(args, A, V_exact, policy):
    """Stream block 1..N_BLOCKS with the given policy.

    policy: callable(mass_shift, ctx) -> 'combined' | 'isvd' | 'oracle'
            ctx is a dict with keys: block, mass_shift, mass_cur, mass_fut

    Returns dict with per-block traces.
    """
    state = None
    half_win = args.half_win
    rank = args.rank
    n = args.n

    cos2 = []
    choice = []
    mass_shift_trace = []
    mass_cur_trace = []
    mass_fut_trace = []
    cos2_block = []  # cos²(state.V[:,0], V_exact[:,0]) right after each block

    for b in range(1, N_BLOCKS + 1):
        sk_end = (b - 1) * half_win
        A_cur = np.asarray(A[sk_end:sk_end + half_win], dtype=np.float32)
        A_fut = np.asarray(A[sk_end + half_win:sk_end + 2 * half_win], dtype=np.float32)

        # Construct M_gain from current state (gated, may differ from pure-combined)
        if state is None:
            B_top = None
            A_sketch = np.zeros((0, A.shape[1]), dtype=np.float32)
            M_gain = A_cur
        else:
            B_top = state["s"].astype(np.float32)[:, None] * state["V"].astype(np.float32).T
            A_sketch = B_top
            M_gain = np.vstack([B_top, A_cur]).astype(np.float32, copy=False)

        # 1. v_cur from combined optimizer (always run; state-dependent)
        V_combined = run_combined_optimizer(M_gain, A_cur, state, rank, args.seed + 7919 * b)
        v_combined = normed(V_combined[:, 0])

        # 2. v_fut for mass-shift signal
        A_cur64 = np.vstack([A_cur, A_fut])
        if B_top is not None:
            M_gain_fut = np.vstack([B_top, A_cur64]).astype(np.float32, copy=False)
        else:
            M_gain_fut = A_cur64
        V_fut = run_combined_optimizer(M_gain_fut, A_cur64, state, rank, args.seed + 7919 * b + 1)
        v_fut = normed(V_fut[:, 0])
        if float(np.dot(v_fut, v_combined)) < 0:
            v_fut = -v_fut

        e_cur = A_cur.astype(np.float64) @ v_combined
        e_fut = A_cur.astype(np.float64) @ v_fut
        mass_cur = float(np.dot(e_cur, e_cur))
        mass_fut = float(np.dot(e_fut, e_fut))
        ms = abs(mass_cur - mass_fut) / max(mass_cur, mass_fut, 1e-30)

        # 3. Alternative slot-1 candidates
        # iSVD: top right SVs of M_gain = [B_top; A_cur]   (matches bench's
        # `policy == "isvd"` branch in half_window_sliding_hmean_experiment.py:494-496)
        Mg64 = M_gain.astype(np.float64)
        _, _, Vt = np.linalg.svd(Mg64, full_matrices=False)
        V_isvd_full = np.ascontiguousarray(Vt[:rank, :].T)
        v_isvd = normed(np.asarray(V_isvd_full[:, 0], dtype=np.float64))
        if float(np.dot(v_isvd, v_combined)) < 0:
            v_isvd = -v_isvd

        # oracle projection: P_M_gain V_exact[:,0]
        B_search = rowspace_basis(Mg64)
        v_oracle = project_unit(V_exact[:, 0].astype(np.float64), B_search)
        if v_oracle is not None and float(np.dot(v_oracle, v_combined)) < 0:
            v_oracle = -v_oracle

        # 4. Pick slot-1 (and slot-2) via policy
        ctx = {"block": b, "mass_shift": ms,
               "mass_cur": mass_cur, "mass_fut": mass_fut,
               "v_oracle_available": v_oracle is not None}
        s1_label, s2_label = policy(ms, ctx)

        def pick_slot(label, slot_idx):
            """slot_idx: 0 = top, 1 = next.
            For 'isvd': top-rank right SVs of M_gain.
            For 'oracle': P_M_gain V_exact[:, slot_idx], renormalized.
            For 'combined': V_combined[:, slot_idx].
            """
            if label == "isvd":
                v = np.asarray(V_isvd_full[:, slot_idx], dtype=np.float64)
            elif label == "oracle":
                if slot_idx >= V_exact.shape[1]:
                    v = None
                else:
                    v = project_unit(V_exact[:, slot_idx].astype(np.float64), B_search)
                if v is None:
                    v = np.asarray(V_isvd_full[:, slot_idx], dtype=np.float64)
            else:  # combined
                v = np.asarray(V_combined[:, slot_idx], dtype=np.float64)
            return normed(v)

        v_slot1 = pick_slot(s1_label, 0)
        if rank >= 2:
            v_slot2 = pick_slot(s2_label, 1)
            v_slot2 = v_slot2 - np.dot(v_slot2, v_slot1) * v_slot1
            v_slot2 = normed(v_slot2)
            V_chosen = np.column_stack([v_slot1, v_slot2])
        else:
            V_chosen = v_slot1.reshape(-1, 1)
        ch = f"{s1_label[0]}{s2_label[0]}"  # 2-char label like "cc", "ic", "oo"

        # 6. Apply make_state
        H_selected = np.zeros(V_chosen.shape[1], dtype=float)
        score_selected = np.zeros(V_chosen.shape[1], dtype=float)
        for j in range(V_chosen.shape[1]):
            sc, _, hh = probe.score_full_vector_details_forget(
                np.asarray(M_gain, dtype=np.float32),
                np.asarray(A_cur, dtype=np.float32),
                np.asarray(V_chosen[:, j], dtype=np.float32),
                A.shape[0],
                state_prev=state, score_variant="combined", old_row_memory=None,
            )
            score_selected[j] = sc
            H_selected[j] = hh
        rows_seen_full = sk_end + half_win
        state, V_r, _ = make_state(np.asarray(M_gain, dtype=np.float32),
                                   np.asarray(V_chosen, dtype=np.float32),
                                   H_selected, score_selected, rows_seen_full)

        v_state_top = np.asarray(state["V"][:, 0], dtype=np.float64)
        c2 = float(np.dot(v_state_top, V_exact[:, 0]) / max(np.linalg.norm(V_exact[:, 0]), 1e-30)) ** 2
        cos2_block.append(c2)
        choice.append(ch)
        mass_shift_trace.append(ms)
        mass_cur_trace.append(mass_cur)
        mass_fut_trace.append(mass_fut)

    return {
        "cos2_block": cos2_block,
        "final_cos2": cos2_block[-1],
        "choice": choice,
        "mass_shift": mass_shift_trace,
        "mass_cur": mass_cur_trace,
        "mass_fut": mass_fut_trace,
    }


def policy_pure(label):
    def fn(ms, ctx):
        return label, label
    return fn


def policy_gated_2way(tau):
    def fn(ms, ctx):
        s1 = "isvd" if ms >= tau else "combined"
        return s1, "combined"  # slot-2 always from combined
    return fn


def policy_gated_3way(tau_isvd, tau_oracle):
    def fn(ms, ctx):
        if ms >= tau_oracle:
            s1 = "oracle"
        elif ms >= tau_isvd:
            s1 = "isvd"
        else:
            s1 = "combined"
        return s1, "combined"
    return fn


POLICY_DEFINITIONS = [
    ("pure_combined",          policy_pure("combined")),
    ("pure_isvd",              policy_pure("isvd")),
    ("pure_oracle",            policy_pure("oracle")),
    ("gated2_tau_0.15",        policy_gated_2way(0.15)),
    ("gated2_tau_0.20",        policy_gated_2way(0.20)),
    ("gated2_tau_0.30",        policy_gated_2way(0.30)),
    ("gated3_0.15_0.35",       policy_gated_3way(0.15, 0.35)),
    ("gated3_0.20_0.40",       policy_gated_3way(0.20, 0.40)),
]


def run_one(matrix, seed):
    args = make_args(seed)
    args.matrix = matrix
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)

    out = {}
    for name, pol in POLICY_DEFINITIONS:
        result = stream_with_policy(args, A, V_exact, pol)
        out[name] = result
        print(f"  {matrix} seed={seed} {name:24s} final cos²={result['final_cos2']:.4f}  "
              f"choices={'/'.join(c[0] for c in result['choice'])}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="summary/mass_shift_gated_stream")
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--quick", action="store_true",
                   help="Run only first matrix × first seed for sanity-check.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.quick:
        run_one(args.matrices[0], args.seeds[0])
        return

    rows = []
    for m in args.matrices:
        for s in args.seeds:
            print(f"=== {m}  seed={s} ===")
            res = run_one(m, s)
            for name, r in res.items():
                rows.append({
                    "matrix": m, "seed": s, "policy": name,
                    "final_cos2": r["final_cos2"],
                    "cos2_block_trace": "|".join(f"{x:.3f}" for x in r["cos2_block"]),
                    "mass_shift_trace": "|".join(f"{x:.3f}" for x in r["mass_shift"]),
                    "choices": "|".join(r["choice"]),
                })

    cells_csv = os.path.join(args.out_dir, "results.csv")
    keys = ["matrix", "seed", "policy", "final_cos2",
            "cos2_block_trace", "mass_shift_trace", "choices"]
    with open(cells_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {cells_csv}")


if __name__ == "__main__":
    main()
