"""Aggregate AB-03 phase 1 (S6 F-norm vs S6_E2 per-direction) bench results.

Reads:
  - this folder's own S6_E2_<matrix>_win64.json (S6_E2 run)
  - ../bench_matrix_sweep_r_sk_g_S6/<matrix>_win64.json (S6_HM3 = F-norm reference)
  - ../bench_matrix_sweep_value_only_online/<matrix>_win64.json (operational online,
    INFRA-10; do NOT use the oracle-aware ../bench_matrix_sweep/)

Emits a side-by-side cos0² / cos1² / tail table and a phase-1 verdict per
the AB-03 acceptance bar derived from DIAG-04b:
  ADVANCE iff S6_E2 improves cos1² on at least one §6 high-entropy
  matrix where DIAG-04b predicted simultaneous u-balance (specifically
  diffuse-diffuse, where E2 achieved ratio_max(slot-2)=2.43×) WITHOUT
  catastrophic regression on the §6 suite.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S6_DIR = ROOT.parent / "bench_matrix_sweep_r_sk_g_S6"
ONLINE_DIR = ROOT.parent / "bench_matrix_sweep_value_only_online"

MATRICES = [
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
]

# DIAG-04b predicted simultaneous oracle-balance gain.
DIAG04B_BALANCE_HIT = {"diffuse-diffuse", "residual-spiky-shocks"}
# DIAG-04b predicted slot-2 regression vs E1.
DIAG04B_SLOT2_REG = {"mixed-tail-sharp"}


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
        e2_p = ROOT / f"S6_E2_{m}_win64.json"
        on_p = ONLINE_DIR / f"{m}_win64.json"

        s6 = load_sliding(s6_p, "future_hmean_r_sk_g")
        e2 = load_sliding(e2_p, "future_hmean_r_sk_g")
        online = load_sliding(on_p, "future_hmean_online")
        isvd = load_sliding(on_p, "isvd")

        rows.append({"matrix": m, "s6": s6, "e2": e2, "online": online, "isvd": isvd})

    print("AB-03 phase 1 (T3) — F-norm vs per-direction-sigma² (S6 vs S6_E2)")
    print("Sliding, half_win=32, block 31. Reference: bench_matrix_sweep_r_sk_g_S6.")
    print("=" * 124)
    print(f"{'matrix':<24} {'S6_HM3 cos0/cos1/tail':<28} {'S6_E2  cos0/cos1/tail':<28} {'online cos0/cos1':<22} {'iSVD cos0/cos1':<22}")
    print("-" * 124)
    for r in rows:
        s6, e2, on, iv = r["s6"], r["e2"], r["online"], r["isvd"]
        s6s = f"{fmt(s6 and s6['cos0'])} {fmt(s6 and s6['cos1'])} {fmt(s6 and s6['tail'])}" if s6 else "  -      -      -   "
        e2s = f"{fmt(e2 and e2['cos0'])} {fmt(e2 and e2['cos1'])} {fmt(e2 and e2['tail'])}" if e2 else "  -      -      -   "
        ons = f"{fmt(on and on['cos0'])} {fmt(on and on['cos1'])}" if on else "  -      -   "
        ivs = f"{fmt(iv and iv['cos0'])} {fmt(iv and iv['cos1'])}" if iv else "  -      -   "
        print(f"{r['matrix']:<24} {s6s:<28} {e2s:<28} {ons:<22} {ivs:<22}")
    print("=" * 124)
    print()

    print("Per-matrix deltas (S6_E2 − S6_HM3):")
    deltas = {}
    for r in rows:
        s6, e2 = r["s6"], r["e2"]
        if s6 and e2:
            d_cos1 = e2["cos1"] - s6["cos1"]
            d_cos0 = e2["cos0"] - s6["cos0"]
            d_tail = e2["tail"] - s6["tail"]
            deltas[r['matrix']] = (d_cos0, d_cos1, d_tail)
            tag = ""
            if r['matrix'] in DIAG04B_BALANCE_HIT:
                tag = "  [DIAG-04b: balance hit]"
            elif r['matrix'] in DIAG04B_SLOT2_REG:
                tag = "  [DIAG-04b: predicted slot-2 regression]"
            print(f"  {r['matrix']:<24}  Δcos0²={d_cos0:+.4f}  Δcos1²={d_cos1:+.4f}  Δtail={d_tail:+.4f}{tag}")

    print()
    print("Acceptance (AB-03 phase 1 T3 ship-screen):")
    diffuse_hit = deltas.get("diffuse-diffuse")
    diffuse_balance_ok = diffuse_hit is not None and diffuse_hit[1] > 0.005
    catastrophic = []
    for m, (d0, d1, _) in deltas.items():
        # Catastrophic = cos1² regression > 0.05 on §6 matrices.
        if d1 < -0.05:
            catastrophic.append((m, d1))
    print(f"  diffuse-diffuse Δcos1² = {diffuse_hit[1]:+.4f}  (DIAG-04b predicted simultaneous balance; need > +0.005)" if diffuse_hit else "  diffuse-diffuse: missing")
    if catastrophic:
        print("  catastrophic regressions (Δcos1² < -0.05):")
        for m, d in catastrophic:
            print(f"    {m}: Δcos1² = {d:+.4f}")
    if diffuse_balance_ok and not catastrophic:
        print("VERDICT: ADVANCE to phase 2 (T2 detailed per-block traces).")
    elif diffuse_balance_ok and catastrophic:
        print("VERDICT: PARTIAL — diffuse hits but catastrophic elsewhere; carry to phase 2 with caveat.")
    else:
        print("VERDICT: KILL — DIAG-04b prediction did not translate to operational cos1² gain.")


if __name__ == "__main__":
    main()
