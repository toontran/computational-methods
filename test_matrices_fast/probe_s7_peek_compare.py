"""Compare S7 with-peek vs no-peek slot-2 picks at b1/b2/b6 on static-cex.

Same streaming carry. At each block:
  with-peek: A_cur=32 rows, A_fut=32 rows, score = (raw_sk+raw_g1+raw_g2)·relH_stacked,
             search = rowspace([sketch; A_cur])
  no-peek:   A_cur=64 rows, A_fut=1 fake row, score = same form,
             search = rowspace([sketch; A_cur_64])  (A_fut is essentially excluded)

Both deflated against V_default[:,0]. Report cos² vs oracle_v1_proj /
oracle_v2_proj, score components, and at divergence point, characterize
the residual direction.
"""

import argparse
import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import stream_to_block, per_block_constants
from r_sk_g_score import optimize_r_sk_g_in_basis


def project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nv = float(np.linalg.norm(p))
    return None if nv <= 1e-30 else p / nv


def cos_sq(a, b):
    if a is None or b is None: return None
    return float(np.dot(a, b)) ** 2


def s7_components(A_sketch, A_cur, A_fut, v):
    eps = 1e-30
    y_sk = A_sketch @ v if A_sketch is not None and A_sketch.size else None
    raw_sk = float(np.dot(y_sk, y_sk)) if y_sk is not None else 0.0
    y_c = A_cur @ v
    raw_g1 = float(np.dot(y_c, y_c))
    y_f = A_fut @ v
    raw_g2 = float(np.dot(y_f, y_f))
    if A_sketch is not None and A_sketch.size:
        M_full = np.vstack([A_sketch, A_cur, A_fut])
    else:
        M_full = np.vstack([A_cur, A_fut])
    yf = M_full @ v
    e = yf * yf
    S = max(float(np.sum(e)), eps)
    p = e / S
    p_pos = np.maximum(p, 1e-300)
    H = -float(np.sum(p * np.log(p_pos)))
    relH = max(H / np.log(max(len(e), 2)), 0.0)
    return {"raw_sk": raw_sk, "raw_g1": raw_g1, "raw_g2": raw_g2,
            "raw_total": raw_sk + raw_g1 + raw_g2, "relH": relH}


def run_s7_pick(A_sketch, A_cur, A_fut, B_search, V_state, c_sk,
                cur_F2, fut_F2, sk_F2_low, seed):
    starts = []
    if V_state is not None and V_state.size:
        for j in range(V_state.shape[1]):
            starts.append(V_state[:, j])
    return optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch, c_sk,
        B_search, starts,
        np.random.default_rng(seed),
        maxit=200, tol=1e-9, random_starts=32,
        variant="S7",
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        V_state=V_state,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="static-cex")
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 6])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32","float64"), default="float32")
    # streaming knobs (passed through to stream_to_block)
    p.add_argument("--q0", type=int, default=8); p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id","U","rand"), default="rand")
    p.add_argument("--out", default="summary/score_family_asym_window/peek_compare_static-cex.txt")
    return p.parse_args()


def main():
    args = parse_args()
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64); V_exact = np.asarray(V_exact, np.float64)
    target = max(args.blocks)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, args.rank, target, set(args.blocks))

    half_win = args.half_win
    lines = []
    def out(s):
        print(s); lines.append(s)
    out(f"== matrix={args.matrix} half_win={half_win} seed={args.seed} ==")
    out(f"Comparing S7 with-peek (32/32) vs no-peek (64/1). Same streaming carry.")
    out("")

    # Cache vectors per block for cross-block analysis
    blocks_data = {}

    for b in sorted(args.blocks):
        if b not in snapshots: continue
        snap = snapshots[b]
        A_cur32 = np.asarray(snap["A_cur"], dtype=np.float64)
        A_fut32 = np.asarray(snap["A_fut"], dtype=np.float64)
        A_sketch = np.asarray(snap["A_sketch"], dtype=np.float64) if snap["A_sketch"].size else np.zeros((0, A.shape[1]), dtype=np.float64)
        V_default = snap["V_default"]
        state = snap["state"]
        V_state = None
        if state is not None and state.get("V") is not None:
            V_state = np.asarray(state["V"], dtype=np.float64)

        # 64-row "no peek" current; 1-row fake peek (for h2>=1 in code)
        A_cur64 = np.vstack([A_cur32, A_fut32])
        # Use a 1-row sample of zeros for "no peek" A_fut (raw_g2 = 0 contribution)
        A_fut1 = np.zeros((1, A.shape[1]), dtype=np.float64)

        consts32 = per_block_constants(A, b, half_win)
        c_sk_32 = consts32["c_sk"]
        cur_F2_32 = float(consts32["cur_F2"])
        fut_F2_32 = float(consts32["fut_F2"])
        # No-peek constants: cur_F2 from 64 rows; fut_F2 from 1-row zero matrix → eps; c_sk same
        cur_F2_64 = float(np.sum(A_cur64 * A_cur64))
        fut_F2_1 = max(float(np.sum(A_fut1 * A_fut1)), 1e-12)
        sk_F2_low = float(np.sum(A_sketch * A_sketch)) if A_sketch.size else 0.0
        c_sk_64 = c_sk_32  # same prefix

        # Search bases
        # with-peek: rowspace([sketch; A_cur32])
        with_peek_stack = np.vstack([A_sketch, A_cur32]) if A_sketch.size else A_cur32
        B_with = rowspace_basis(with_peek_stack)
        # no-peek: rowspace([sketch; A_cur64])
        no_peek_stack = np.vstack([A_sketch, A_cur64]) if A_sketch.size else A_cur64
        B_no = rowspace_basis(no_peek_stack)

        # Deflate against V_default[:, 0]
        if V_default is not None and V_default.shape[1] >= 1:
            v1_anchor = V_default[:, 0]
            B_with_def = orth_basis_against(B_with, v1_anchor)
            B_no_def = orth_basis_against(B_no, v1_anchor)
        else:
            B_with_def = B_with; B_no_def = B_no

        # Oracle projections: project into rowspace of full [sketch; A_cur32; A_fut32]
        union = np.vstack([A_sketch, A_cur32, A_fut32]) if A_sketch.size else np.vstack([A_cur32, A_fut32])
        B_union = rowspace_basis(union)
        o1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
        o2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
        o1p = project_unit(o1, B_union)
        o2p = project_unit(o2, B_union)

        # Run optimizers (slot-2 picks)
        wp_best = run_s7_pick(A_sketch if A_sketch.size else None, A_cur32, A_fut32,
                              B_with_def, V_state, c_sk_32,
                              cur_F2_32, fut_F2_32, sk_F2_low, args.seed + 100*b)
        np_best = run_s7_pick(A_sketch if A_sketch.size else None, A_cur64, A_fut1,
                              B_no_def, V_state, c_sk_64,
                              cur_F2_64, fut_F2_1, sk_F2_low, args.seed + 100*b + 1)

        v_wp = wp_best["vec"] if wp_best is not None else None
        v_np = np_best["vec"] if np_best is not None else None
        if v_wp is not None: v_wp = v_wp / max(np.linalg.norm(v_wp), 1e-30)
        if v_np is not None: v_np = v_np / max(np.linalg.norm(v_np), 1e-30)

        comp_wp = s7_components(A_sketch if A_sketch.size else None, A_cur32, A_fut32, v_wp) if v_wp is not None else None
        # For no-peek, recompute components in BOTH framings:
        #   (a) the 64/1 framing (the score the optimizer used)
        comp_np_64 = s7_components(A_sketch if A_sketch.size else None, A_cur64, A_fut1, v_np) if v_np is not None else None
        #   (b) the 32/32 framing (so the comparison vs comp_wp is on the same basis)
        comp_np_32 = s7_components(A_sketch if A_sketch.size else None, A_cur32, A_fut32, v_np) if v_np is not None else None

        # Cosines
        wp_o1 = cos_sq(v_wp, o1p) if v_wp is not None else None
        wp_o2 = cos_sq(v_wp, o2p) if v_wp is not None else None
        np_o1 = cos_sq(v_np, o1p) if v_np is not None else None
        np_o2 = cos_sq(v_np, o2p) if v_np is not None else None
        wp_np = cos_sq(v_wp, v_np) if (v_wp is not None and v_np is not None) else None

        # Project oracle_v?_proj into the with-peek basis to see what's reachable
        def s7_score_full(A_sk, A_c, A_f, v):
            c = s7_components(A_sk if A_sk is not None and A_sk.size else None, A_c, A_f, v)
            return c["raw_total"] * c["relH"], c

        def proj_to(B, vec):
            if B is None or B.size == 0 or vec is None: return None
            p = B @ (B.T @ vec)
            n = float(np.linalg.norm(p))
            return None if n <= 1e-30 else p / n

        # Reachable oracle in with-peek basis (deflated):
        o1_wp = proj_to(B_with_def, o1p) if o1p is not None else None
        o2_wp = proj_to(B_with_def, o2p) if o2p is not None else None
        # Reachable oracle in no-peek basis (deflated):
        o1_np = proj_to(B_no_def, o1p) if o1p is not None else None
        o2_np = proj_to(B_no_def, o2p) if o2p is not None else None

        out(f"-- block {b}  N_sk={A_sketch.shape[0]}  cur_F2(32)={cur_F2_32:.3f}  fut_F2(32)={fut_F2_32:.3f}  sk_F2_low={sk_F2_low:.3f} --")
        # How much of the oracle survives the deflated with-peek projection?
        if o2p is not None:
            o2_wp_pre_norm = float(np.linalg.norm(B_with_def @ (B_with_def.T @ o2p)))
            o2_np_pre_norm = float(np.linalg.norm(B_no_def @ (B_no_def.T @ o2p)))
            out(f"  reachable mass: ||proj_B_with_def(oracle_v2_proj)||={o2_wp_pre_norm:.4f}  ||proj_B_no_def(oracle_v2_proj)||={o2_np_pre_norm:.4f}")
            o1_wp_pre_norm = float(np.linalg.norm(B_with_def @ (B_with_def.T @ o1p)))
            o1_np_pre_norm = float(np.linalg.norm(B_no_def @ (B_no_def.T @ o1p)))
            out(f"  reachable mass: ||proj_B_with_def(oracle_v1_proj)||={o1_wp_pre_norm:.4f}  ||proj_B_no_def(oracle_v1_proj)||={o1_np_pre_norm:.4f}")
        # FULL oracle (in union basis, NOT projected to with-peek) — so we see
        # its raw_g1/raw_g2 with both halves intact
        if o2p is not None:
            sf, c2f = s7_score_full(A_sketch, A_cur32, A_fut32, o2p)
            out(f"  S7 score @ oracle_v2_proj (full, in B_union):    {sf:.5e}  raw_g1={c2f['raw_g1']:.4e}  raw_g2={c2f['raw_g2']:.4e}  relH={c2f['relH']:.4f}")
        if o1p is not None:
            sf, c1f = s7_score_full(A_sketch, A_cur32, A_fut32, o1p)
            out(f"  S7 score @ oracle_v1_proj (full, in B_union):    {sf:.5e}  raw_g1={c1f['raw_g1']:.4e}  raw_g2={c1f['raw_g2']:.4e}  relH={c1f['relH']:.4f}")
        # Cross-window orthogonality probe: principal angles cos² between
        # rowspace(A_cur32) and rowspace(A_fut32), top 5
        Q_cur = rowspace_basis(A_cur32)
        Q_fut = rowspace_basis(A_fut32)
        if Q_cur.size and Q_fut.size:
            from numpy.linalg import svd as _svd
            sigmas = _svd(Q_cur.T @ Q_fut, compute_uv=False)
            top = " ".join(f"{s*s:.3f}" for s in sigmas[:5])
            out(f"  rowspace(A_cur) vs rowspace(A_fut): top-5 principal cos²: {top}  (max={sigmas[0]**2:.3f})")
        # S7 with-peek score evaluated at the oracle-projected reachable directions
        if o2_wp is not None:
            sw, c2w = s7_score_full(A_sketch, A_cur32, A_fut32, o2_wp)
            out(f"  S7 score @ oracle_v2_proj projected→B_with_def: {sw:.5e}  raw_g1={c2w['raw_g1']:.4e}  raw_g2={c2w['raw_g2']:.4e}  relH={c2w['relH']:.4f}")
        if o1_wp is not None:
            sw, c1w = s7_score_full(A_sketch, A_cur32, A_fut32, o1_wp)
            out(f"  S7 score @ oracle_v1_proj projected→B_with_def: {sw:.5e}  raw_g1={c1w['raw_g1']:.4e}  raw_g2={c1w['raw_g2']:.4e}  relH={c1w['relH']:.4f}")
        # S7 no-peek score evaluated at oracle projections (in 64/1 frame)
        if o2_np is not None:
            sn, c2n = s7_score_full(A_sketch, A_cur64, A_fut1, o2_np)
            out(f"  S7 score @ oracle_v2_proj projected→B_no_def  (64/1 frame): {sn:.5e}  raw_g1={c2n['raw_g1']:.4e}  relH={c2n['relH']:.4f}")
        if o1_np is not None:
            sn, c1n = s7_score_full(A_sketch, A_cur64, A_fut1, o1_np)
            out(f"  S7 score @ oracle_v1_proj projected→B_no_def  (64/1 frame): {sn:.5e}  raw_g1={c1n['raw_g1']:.4e}  relH={c1n['relH']:.4f}")
        out(f"  Anchor V_default[:,0] cos²(o1p)={cos_sq(V_default[:,0]/max(np.linalg.norm(V_default[:,0]),1e-30), o1p):.4f}  cos²(o2p)={cos_sq(V_default[:,0]/max(np.linalg.norm(V_default[:,0]),1e-30), o2p):.4f}")
        # First 5 elements of v_wp / v_np for v_type=id debugging (oracle_vk = e_{k-1})
        if v_wp is not None:
            head = " ".join(f"{x:+.3f}" for x in v_wp[:5])
            top_idx = np.argsort(np.abs(v_wp))[::-1][:5]
            top = ", ".join(f"v[{i}]={v_wp[i]:+.3f}" for i in top_idx)
            out(f"  v_wp[:5]={head}  top5_by_|v|: {top}")
        if v_np is not None:
            head = " ".join(f"{x:+.3f}" for x in v_np[:5])
            top_idx = np.argsort(np.abs(v_np))[::-1][:5]
            top = ", ".join(f"v[{i}]={v_np[i]:+.3f}" for i in top_idx)
            out(f"  v_np[:5]={head}  top5_by_|v|: {top}")
        out(f"  with-peek v2:  score={wp_best['score']:.5e}  cos²(o1p)={wp_o1:.4f}  cos²(o2p)={wp_o2:.4f}")
        out(f"   components(32/32 frame): raw_sk={comp_wp['raw_sk']:.4e}  raw_g1={comp_wp['raw_g1']:.4e}  raw_g2={comp_wp['raw_g2']:.4e}  relH={comp_wp['relH']:.4f}  raw_total={comp_wp['raw_total']:.4e}")
        out(f"  no-peek  v2:   score={np_best['score']:.5e}  cos²(o1p)={np_o1:.4f}  cos²(o2p)={np_o2:.4f}")
        out(f"   components(64/1 frame):  raw_sk={comp_np_64['raw_sk']:.4e}  raw_g1={comp_np_64['raw_g1']:.4e}  raw_g2={comp_np_64['raw_g2']:.4e}  relH={comp_np_64['relH']:.4f}")
        out(f"   components(32/32 frame): raw_sk={comp_np_32['raw_sk']:.4e}  raw_g1={comp_np_32['raw_g1']:.4e}  raw_g2={comp_np_32['raw_g2']:.4e}  relH={comp_np_32['relH']:.4f}")
        out(f"  cos²(v_wp, v_np) = {wp_np:.4f}")

        blocks_data[b] = {"v_wp": v_wp, "v_np": v_np, "o1p": o1p, "o2p": o2p,
                          "V_state": V_state, "A_sketch": A_sketch,
                          "A_cur32": A_cur32, "A_fut32": A_fut32,
                          "V_default": V_default}
        out("")

    # Find divergence block: cos²(v_wp, v_np) drops below ~0.95
    divergence_block = None
    for b in sorted(blocks_data.keys()):
        v_wp = blocks_data[b]["v_wp"]; v_np = blocks_data[b]["v_np"]
        if v_wp is None or v_np is None: continue
        c = cos_sq(v_wp, v_np)
        if c < 0.95:
            divergence_block = b; break

    if divergence_block is None:
        out("No divergence (cos² >= 0.95 at all probed blocks).")
    else:
        out(f"== Divergence block: {divergence_block} ==")
        bd = blocks_data[divergence_block]
        v_wp = bd["v_wp"]; v_np = bd["v_np"]
        # residual = v_wp - <v_wp, v_np> v_np  (orthogonal complement of v_wp wrt v_np)
        proj_coef = float(np.dot(v_wp, v_np))
        residual = v_wp - proj_coef * v_np
        rn = float(np.linalg.norm(residual))
        out(f"  ||residual|| = {rn:.4f}  (residual = v_wp - <v_wp,v_np> v_np)")
        if rn > 1e-10:
            r_unit = residual / rn
            # Cos² vs oracles (raw, not projected)
            o1raw = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
            o2raw = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
            out(f"  cos²(residual_unit, V_exact[:,0]) = {cos_sq(r_unit, o1raw):.4f}")
            out(f"  cos²(residual_unit, V_exact[:,1]) = {cos_sq(r_unit, o2raw):.4f}")
            o1p = bd["o1p"]; o2p = bd["o2p"]
            out(f"  cos²(residual_unit, oracle_v1_proj) = {cos_sq(r_unit, o1p):.4f}")
            out(f"  cos²(residual_unit, oracle_v2_proj) = {cos_sq(r_unit, o2p):.4f}")
            # Cos² vs state.V columns
            V_state = bd["V_state"]
            if V_state is not None and V_state.size:
                for j in range(V_state.shape[1]):
                    sv = V_state[:, j] / max(np.linalg.norm(V_state[:, j]), 1e-30)
                    out(f"  cos²(residual_unit, state.V[:,{j}]) = {cos_sq(r_unit, sv):.4f}")
            # Top-5 cos² of residual against rows of A_cur32, A_fut32
            for label, M in [("A_cur32", bd["A_cur32"]), ("A_fut32", bd["A_fut32"]), ("A_sketch", bd["A_sketch"])]:
                if M.size == 0: continue
                proj = M @ r_unit
                norms = np.linalg.norm(M, axis=1) + 1e-30
                cos2 = (proj * proj) / (norms * norms)
                idx = np.argsort(cos2)[::-1][:5]
                top = ", ".join(f"row{i}={cos2[i]:.3f}" for i in idx)
                out(f"  top-5 cos²(residual_unit, {label} rows): {top}")
            # Cos² of residual against V_exact columns 2..6
            for j in range(2, min(6, V_exact.shape[1])):
                ojr = V_exact[:, j] / max(np.linalg.norm(V_exact[:, j]), 1e-30)
                out(f"  cos²(residual_unit, V_exact[:,{j}]) = {cos_sq(r_unit, ojr):.4f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
