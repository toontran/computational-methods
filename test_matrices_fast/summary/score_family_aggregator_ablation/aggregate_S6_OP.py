"""Aggregate AB-02 (F-norm vs OP-norm) bench results.

Reads:
  - this folder's own *_S6_OP_win64.json (S6_OP run)
  - ../bench_matrix_sweep_r_sk_g_S6/*_win64.json (S6_HM3 = F-norm reference)
  - ../bench_matrix_sweep_value_only_online/*_win64.json (operational online,
    INFRA-10; do NOT use the oracle-aware ../bench_matrix_sweep/)

Emits a side-by-side cos0² / cos1² / tail table and a within-family
SHIP/KILL verdict per the AB-02 acceptance bar:
  SHIP S6_OP iff it improves cos1² on heavy-tailed matrices (specifically
  static-cex) WITHOUT regressing diffuse — i.e. op-norm weighting earns a
  propagation to FAM-01.
  KILL otherwise.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S6_DIR = ROOT.parent / "bench_matrix_sweep_r_sk_g_S6"
ONLINE_DIR = ROOT.parent / "bench_matrix_sweep_value_only_online"

# §6 matrices (operational target).
MATRICES = [
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
]

# Heavy-tailed subset (the AB-02 acceptance asks specifically about
# static-cex; mixed-tail-* are also heavy-tailed in the §6 sense).
HEAVY_TAILED = {"static-cex", "mixed-tail-sharp", "mixed-tail-balanced",
                "mixed-tail-soft"}
# Diffuse subset (must not regress).
DIFFUSE = {"diffuse-diffuse"}


def load_sliding(path, policy):
    if not path.exists():
        return None
    with open(path) as fp:
        d = json.load(fp)
    for s in d["summaries"]:
        if s["mode"] == "sliding" and s["policy"] == policy:
            cos = s.get("final_exact_cos") or [None, None]
            return {
                "cos0": float(cos[0]) if cos and cos[0] is not None else None,
                "cos1": float(cos[1]) if len(cos) > 1 and cos[1] is not None else None,
                "tail": float(s.get("final_tail_mass") or float("nan")),
            }
    return None


def fmt(x):
    if x is None:
        return "  -   "
    return f"{x:.4f}"


def main():
    rows = []
    for m in MATRICES:
        s6_p = S6_DIR / f"{m}_win64.json"
        op_p = ROOT / f"{m}_S6_OP_win64.json"
        on_p = ONLINE_DIR / f"{m}_win64.json"

        s6 = load_sliding(s6_p, "future_hmean_r_sk_g")
        op = load_sliding(op_p, "future_hmean_r_sk_g")
        online = load_sliding(on_p, "future_hmean_online")
        isvd = load_sliding(on_p, "isvd")

        rows.append({
            "matrix": m,
            "s6": s6,
            "op": op,
            "online": online,
            "isvd": isvd,
        })

    print("AB-02 F-norm vs OP-norm aggregator ablation — side-by-side cos² (sliding, half_win=32, block 31)")
    print("(S6_HM3 = F-norm baseline; online = value-only-online INFRA-10; do NOT compare against the oracle-aware ../bench_matrix_sweep/)")
    print("=" * 120)
    print(f"{'matrix':<24} {'S6_HM3 cos0/cos1/tail':<28} {'S6_OP  cos0/cos1/tail':<28} {'online cos0/cos1':<22} {'iSVD cos0/cos1':<22}")
    print("-" * 120)
    for r in rows:
        s6, op, on, iv = r["s6"], r["op"], r["online"], r["isvd"]
        s6s = f"{fmt(s6 and s6['cos0'])} {fmt(s6 and s6['cos1'])} {fmt(s6 and s6['tail'])}" if s6 else "  -      -      -   "
        ops = f"{fmt(op and op['cos0'])} {fmt(op and op['cos1'])} {fmt(op and op['tail'])}" if op else "  -      -      -   "
        ons = f"{fmt(on and on['cos0'])} {fmt(on and on['cos1'])}" if on else "  -      -   "
        ivs = f"{fmt(iv and iv['cos0'])} {fmt(iv and iv['cos1'])}" if iv else "  -      -   "
        print(f"{r['matrix']:<24} {s6s:<28} {ops:<28} {ons:<22} {ivs:<22}")
    print("=" * 120)
    print()

    print("Per-matrix cos1² deltas (S6_OP − S6_HM3):")
    static_cex_delta = None
    diffuse_cos1_delta = None
    diffuse_cos0_delta = None
    for r in rows:
        s6, op = r["s6"], r["op"]
        if s6 and op:
            d_cos1 = op["cos1"] - s6["cos1"]
            d_cos0 = op["cos0"] - s6["cos0"]
            sign1 = "+" if d_cos1 > 0 else ""
            sign0 = "+" if d_cos0 > 0 else ""
            print(f"  {r['matrix']:<24} S6={s6['cos1']:.4f}  S6_OP={op['cos1']:.4f}  Δcos1²={sign1}{d_cos1:+.4f}   Δcos0²={sign0}{d_cos0:+.4f}")
            if r['matrix'] == 'static-cex':
                static_cex_delta = d_cos1
            if r['matrix'] == 'diffuse-diffuse':
                diffuse_cos1_delta = d_cos1
                diffuse_cos0_delta = d_cos0

    print()
    print("AB-02 ship rule: SHIP iff S6_OP improves cos1² on static-cex WITHOUT regressing diffuse-diffuse (cos0² and cos1²).")
    print(f"  static-cex Δcos1²   = {static_cex_delta:+.4f}" if static_cex_delta is not None else "  static-cex: missing")
    print(f"  diffuse Δcos1²      = {diffuse_cos1_delta:+.4f}" if diffuse_cos1_delta is not None else "  diffuse: missing")
    print(f"  diffuse Δcos0²      = {diffuse_cos0_delta:+.4f}" if diffuse_cos0_delta is not None else "  diffuse: missing")

    static_improves = static_cex_delta is not None and static_cex_delta > 0.005  # >0.5% cos1²
    diffuse_no_regress = (
        diffuse_cos1_delta is not None and diffuse_cos1_delta >= -0.01 and
        diffuse_cos0_delta is not None and diffuse_cos0_delta >= -0.01
    )
    if static_improves and diffuse_no_regress:
        print("VERDICT: SHIP")
    else:
        print("VERDICT: KILL")
        if not static_improves:
            print(f"  - static-cex did not improve (Δcos1² = {static_cex_delta:+.4f}; need > +0.005)")
        if not diffuse_no_regress:
            print(f"  - diffuse regressed (cos0² Δ={diffuse_cos0_delta:+.4f}, cos1² Δ={diffuse_cos1_delta:+.4f})")

    # Side check: did S6_OP help on the high-entropy failures of S6?
    # mixed-tail-sharp 0.013, mixed-tail-balanced 0.0004, diffuse-diffuse 0.005 in §6/value-only-online
    print()
    print("Side check — high-entropy regime (where S6 is operationally weakest):")
    for m in ["mixed-tail-sharp", "mixed-tail-balanced", "diffuse-diffuse"]:
        for r in rows:
            if r["matrix"] == m and r["s6"] and r["op"]:
                d_cos1 = r["op"]["cos1"] - r["s6"]["cos1"]
                d_cos0 = r["op"]["cos0"] - r["s6"]["cos0"]
                print(f"  {m:<24} cos1²: S6={r['s6']['cos1']:.4f}  S6_OP={r['op']['cos1']:.4f}  Δ={d_cos1:+.4f} | cos0²: Δ={d_cos0:+.4f}")


if __name__ == "__main__":
    main()
