"""Aggregate AB-01 (HM vs GM) bench results.

Reads:
  - this folder's own *_win64.json (S6_GM run, plus combined / isvd baselines)
  - ../bench_matrix_sweep_r_sk_g_S6/*_win64.json (S6 reference)
  - ../bench_matrix_sweep/*_win64.json (online + iSVD reference)
  - ../benchmark_online_vs_baselines_win64.json (mixed-tail-sharp solo run)

Emits a side-by-side cos0² / cos1² / tail table with the §6 matrices, and a
ship/kill verdict per the AB-01 acceptance bar:
  Ship if GM closes >=50% of the S6→online gap on any tail-dominant matrix.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S6_DIR = ROOT.parent / "bench_matrix_sweep_r_sk_g_S6"
ONLINE_DIR = ROOT.parent / "bench_matrix_sweep"
ONLINE_SHARP = ROOT.parent / "benchmark_online_vs_baselines_win64.json"

# Matrices in the §6 table of score_design_overview.txt (operational target).
MATRICES = [
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
]

# Tail-dominant subset for the ship/kill verdict.
TAIL_DOMINANT = {
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "diffuse-diffuse",
    "etf-basket-basis",
    "mixed-tail-soft",
}


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


def online_path(matrix):
    if matrix == "mixed-tail-sharp" and ONLINE_SHARP.exists():
        return ONLINE_SHARP
    return ONLINE_DIR / f"{matrix}_win64.json"


def main():
    rows = []
    for m in MATRICES:
        s6_p = S6_DIR / f"{m}_win64.json"
        gm_p = ROOT / f"{m}_win64.json"
        on_p = online_path(m)

        s6 = load_sliding(s6_p, "future_hmean_r_sk_g")
        gm = load_sliding(gm_p, "future_hmean_r_sk_g")
        online = load_sliding(on_p, "future_hmean_online")
        isvd = load_sliding(on_p, "isvd")

        rows.append({
            "matrix": m,
            "s6": s6,
            "gm": gm,
            "online": online,
            "isvd": isvd,
        })

    print("AB-01 HM-vs-GM aggregator ablation — side-by-side cos² (sliding, half_win=32, block 31)")
    print("=" * 110)
    print(f"{'matrix':<24} {'S6_HM3 cos0/cos1/tail':<28} {'S6_GM  cos0/cos1/tail':<28} {'online cos0/cos1':<22} {'iSVD cos0/cos1':<22}")
    print("-" * 110)
    ship = False
    ship_reasons = []
    for r in rows:
        s6, gm, on, iv = r["s6"], r["gm"], r["online"], r["isvd"]
        s6s = f"{fmt(s6 and s6['cos0'])} {fmt(s6 and s6['cos1'])} {fmt(s6 and s6['tail'])}" if s6 else "  -      -      -   "
        gms = f"{fmt(gm and gm['cos0'])} {fmt(gm and gm['cos1'])} {fmt(gm and gm['tail'])}" if gm else "  -      -      -   "
        ons = f"{fmt(on and on['cos0'])} {fmt(on and on['cos1'])}" if on else "  -      -   "
        ivs = f"{fmt(iv and iv['cos0'])} {fmt(iv and iv['cos1'])}" if iv else "  -      -   "
        print(f"{r['matrix']:<24} {s6s:<28} {gms:<28} {ons:<22} {ivs:<22}")

        # Ship-rule: GM closes >=50% of S6->online gap on any tail-dominant matrix.
        if r['matrix'] in TAIL_DOMINANT and s6 and gm and on:
            gap_s6 = on['cos1'] - s6['cos1']
            gap_gm = on['cos1'] - gm['cos1']
            if gap_s6 > 0.0:
                closed = (gap_s6 - gap_gm) / gap_s6
                if closed >= 0.5:
                    ship = True
                    ship_reasons.append(f"{r['matrix']}: GM closed {closed:.0%} (S6={s6['cos1']:.3f}, GM={gm['cos1']:.3f}, online={on['cos1']:.3f})")

    print("=" * 110)
    print()
    print("Ship/kill rule: ship iff GM closes >=50% of the S6->online cos1² gap on ANY tail-dominant matrix.")
    if ship:
        print("VERDICT: SHIP")
        for r in ship_reasons:
            print("  +", r)
    else:
        print("VERDICT: KILL")
        print("  No tail-dominant matrix where GM closes >=50% of the S6->online gap.")

    # Per-matrix delta summary for the synthesis.
    print()
    print("Per-matrix cos1² deltas (GM − S6_HM3):")
    for r in rows:
        s6, gm = r["s6"], r["gm"]
        if s6 and gm:
            d = gm["cos1"] - s6["cos1"]
            sign = "+" if d > 0 else ""
            print(f"  {r['matrix']:<24} S6={s6['cos1']:.4f}  GM={gm['cos1']:.4f}  Δ={sign}{d:+.4f}")


if __name__ == "__main__":
    main()
