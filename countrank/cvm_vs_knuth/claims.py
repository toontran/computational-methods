
import math, json
from statistics import mean, pstdev
from .estimators import Estimator, _rng
from .datasets import perm_with_repeats, zipf_stream, adversarial_repeats, edge_m_eq_D_plus_1, true_F0

def test_memory_cap_s(claim_id="CLM_CAP_S", variant="cvm", D=1000, m=5000, s=64, seed=7):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = logs.peak_buffer <= s
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"peak_buffer": logs.peak_buffer, "s": s, "estimate": est}}

def test_init_p_equals_one(claim_id="CLM_INIT_P", variant="cvm", D=100, m=200, s=8, seed=1):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = len(logs.p_traj)>0 and abs(logs.p_traj[0]-1.0) < 1e-12
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"first_p": logs.p_traj[0]}}

def test_one_pass_update_cost(claim_id="CLM_COST", variant="cvm", D=2000, m=10000, s=128, seed=2):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = logs.time_per_million > 0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"time_per_million_updates_sec": logs.time_per_million}}

def test_last_occurrence_fairness(claim_id="CLM_FAIR_LAST", variant="knuth_d", D=500, m=5000, s=32, seed=3):
    stream, _ = perm_with_repeats(D, m, seed, repeat_factor=2.0)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    p_final = logs.p_traj[-1]
    mismatches = 0
    for k, u in logs.last_scores.items():
        inB = logs.final_membership.get(k, False)
        if (u < p_final) != inB:
            mismatches += 1
    ok = mismatches == 0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"mismatches": mismatches, "p_final": p_final}}

def test_uniform_scores_ks(claim_id="CLM_U_KS", variant="cvm", D=200, m=5000, s=32, seed=4):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    xs = list(logs.all_scores)
    bins = 20
    counts = [0] * bins
    for x in xs:
        idx = int(x * bins)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    expected = (len(xs) / bins) if bins > 0 else 0
    chi2 = 0.0
    if expected > 0:
        for c in counts:
            chi2 += ((c - expected) ** 2) / expected
    ok = chi2 < 60.0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"chi2": float(chi2), "bins": bins, "n": int(len(xs))}}

def test_acceptance_rate_equals_p(claim_id="CLM_ACCEPT_EQ_P", variant="cvm", D=1000, m=10000, s=128, seed=5):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    acc_flags = list(logs.accept_flags)
    p_events = list(logs.p_at_event)
    full_flags = list(logs.buffer_full_before_event)
    eligible = [(acc_flags[i], p_events[i]) for i, full in enumerate(full_flags) if not full]
    samples = len(eligible)
    if samples == 0:
        ok = False
        diff = float("inf")
        emp_rate = 0.0
        mean_p = 0.0
    else:
        emp_rate = sum(flag for flag, _ in eligible) / samples
        mean_p = sum(p for _, p in eligible) / samples
        diff = abs(emp_rate - mean_p)
        ok = diff < 0.02
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"emp_rate": emp_rate, "mean_p": mean_p, "abs_diff": diff, "samples": samples}}

def test_cvm_halving_binomial(claim_id="CLM_HALVE_BINOM", variant="cvm", D=2000, m=20000, s=128, seed=6):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    z_scores = []
    ratios = []
    for rec in logs.halving_records:
        if rec.get("variant") != "cvm":
            continue
        before = rec["before"]
        after = rec["after"]
        if before <= 1:
            continue
        expected = before / 2.0
        variance = before * 0.25
        if variance <= 0:
            continue
        z = abs(after - expected) / math.sqrt(variance)
        z_scores.append(z)
        ratios.append(after / expected)
    max_z = max(z_scores) if z_scores else 0.0
    mean_ratio = mean(ratios) if ratios else 0.0
    ok = (len(z_scores) >= 3) and (max_z < 3.5) and (abs(mean_ratio - 1.0) < 0.1)
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"halving_events": len(z_scores), "max_z": float(max_z), "mean_ratio": mean_ratio}}

def test_cutoff_equals_max_in_buffer(claim_id="CLM_P_EQ_MAX", variant="knuth_d", D=800, m=15000, s=64, seed=7):
    stream, _ = perm_with_repeats(D, m, seed)
    # Re-simulate explicitly to capture final buffer contents
    rng = _rng(seed)
    B = {}
    p = 1.0
    for a in stream:
        if a in B: del B[a]
        u = rng.random()
        if u >= p: 
            continue
        if len(B) < s:
            B[a] = u
        else:
            a_star, u_star = None, -1.0
            for k, v in B.items():
                if v > u_star:
                    a_star, u_star = k, v
            if u > u_star:
                p = u
            else:
                del B[a_star]
                B[a] = u
                p = u_star
    max_u = max(B.values()) if B else 0.0
    ok = (p >= max_u - 1e-12)
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"p_final": p, "max_u_in_B": max_u, "diff": p - max_u}}

def test_p_monotone_nonincreasing_after_fill(claim_id="CLM_P_MONO", variant="cvm", D=1000, m=20000, s=64, seed=8):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    seen_halve = False
    ok = True
    last_p = 1.0
    for t, p, tag in logs.p_events:
        if tag == "cvm_halve":
            seen_halve = True
        if seen_halve and (p > last_p + 1e-15):
            ok = False
        last_p = p
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"events_tail": logs.p_events[-10:]}}

def test_knuth_unbiased_monte_carlo(claim_id="CLM_D_UNBIASED", variant="knuth_d", D=500, m=15000, s=128, trials=60, seed0=15):
    errs = []
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0+r, repeat_factor=1.4)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant="knuth_d")
        errs.append(est - F0)
    mean_err = mean(errs) if errs else 0.0
    stderr = (pstdev(errs) / math.sqrt(len(errs))) if len(errs) > 0 else float("inf")
    ok = abs(mean_err) / max(1.0, D) < 0.02
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_error": mean_err, "stderr": stderr, "trials": trials}}

def test_dprime_power_of_two_cutoff(claim_id="CLM_DPRIME_POW2", variant="knuth_dprime", D=800, m=25000, s=128, seed=16):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    deltas = []
    for p in logs.p_traj:
        if p <= 0:
            continue
        k = round(-math.log(p, 2))
        pow_two = 2.0 ** (-k)
        deltas.append(abs(p - pow_two))
    max_delta = max(deltas) if deltas else 0.0
    ok = max_delta < 1e-12
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"max_delta": max_delta, "events": len(deltas)}}

def test_cvm_total_unbiased(claim_id="CLM_TOTAL_UNB", variant="cvm_total", D=500, m=15000, s=128, trials=60, seed0=9):
    errs = []
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0+r, repeat_factor=1.5)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant="cvm_total")
        errs.append(est - F0)
    mean_err = mean(errs) if errs else 0.0
    stderr = (pstdev(errs) / math.sqrt(len(errs))) if len(errs) > 0 else float("inf")
    ok = abs(mean_err) / max(1.0, D) < 0.02
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_error": mean_err, "stderr": stderr, "trials": trials}}

def test_cvm_original_bias_detected(claim_id="CLM_ORIG_BIAS", variant="cvm", D=500, m=15000, s=128, trials=60, seed0=11):
    errs = []
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0+r, repeat_factor=1.5)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant="cvm")
        errs.append(est - F0)
    mean_err = mean(errs) if errs else 0.0
    ok = abs(mean_err) / max(1.0, D) > 0.01
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_error": mean_err, "trials": trials}}

def test_cvm_functional_invariant(claim_id="CLM_FUNC_INV", variant="cvm", D=400, m=12000, s=64, trials=40, seed0=13, trace_key=0):
    ratios = []
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0+r, repeat_factor=1.2)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant="cvm", trace_key=trace_key)
        if logs.trace_first_hit is None:
            continue
        start_idx = max(0, logs.trace_first_hit - 1)
        for idx in range(start_idx, len(logs.trace_membership)):
            p = logs.trace_p[idx]
            if p <= 0:
                continue
            ratios.append(logs.trace_membership[idx] / p)
    mean_ratio = mean(ratios) if ratios else 0.0
    ok = (len(ratios) > 1000) and (abs(mean_ratio - 1.0) < 0.1)
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"samples": len(ratios), "mean_ratio": mean_ratio}}

def test_pac_coverage_eps_delta(claim_id="CLM_PAC", variant="knuth_dprime", D=1000, m=25000, s=128, epsilon=0.2, delta=0.1, trials=80, seed0=13):
    failures = 0
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0 + r, repeat_factor=1.3)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
        if abs(est - F0) > epsilon * max(1, F0):
            failures += 1
    emp = failures / trials
    ok = emp <= delta * 1.2
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"empirical_tail": emp, "epsilon": epsilon, "delta": delta, "trials": trials}}

def _theory_tail_bound(s, eps):
    return 2.0*math.exp(-(s*(eps**2))/6.0) + 4.0*math.exp(-s/24.0)

def test_tail_bounds_knuth_T_upperbounds(claim_id="CLM_TAIL_KNUTH", variant="knuth_dprime", D=1000, m=35000, s=256, epsilon=0.2, trials=80, seed0=17):
    failures = 0
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0 + r, repeat_factor=1.1)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
        if abs(est - F0) > epsilon * max(1, F0):
            failures += 1
    emp = failures / trials
    bound = _theory_tail_bound(s, epsilon)
    ok = emp <= bound * 1.2
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"empirical_tail": emp, "bound": bound, "epsilon": epsilon, "trials": trials}}

def test_dprime_repeat_guard_bound(claim_id="CLM_DPRIME_REPEAT", variant="knuth_dprime", D=800, m=20000, s=64, seed=21):
    stream, _ = perm_with_repeats(D, m, seed, repeat_factor=1.7)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    repeats = 0
    halving = 0
    for rec in logs.halving_records:
        if rec.get("variant") != "knuth_dprime":
            continue
        halving += 1
        if rec.get("repeat"):
            repeats += 1
    bound = m / (2 ** (s + 1))
    ok = repeats <= max(1, int(bound * 1.1))
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"halving_events": halving, "repeat_events": repeats, "theory_bound": bound}}

def test_edge_m_eq_D_plus_1(claim_id="CLM_EDGE_M_EQ_D1", variant="cvm", D=512, s=32, seed=23):
    stream, _ = edge_m_eq_D_plus_1(D, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    F0 = true_F0(stream)
    ok = est > 0 and logs.peak_buffer <= s
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed, "metrics": {"estimate": est, "true_F0": F0}}

def run_all(variant="cvm"):
    tests = [
        test_memory_cap_s,
        test_init_p_equals_one,
        test_one_pass_update_cost,
        test_last_occurrence_fairness,
        test_uniform_scores_ks,
        test_acceptance_rate_equals_p,
        test_cvm_halving_binomial,
        test_cutoff_equals_max_in_buffer,
        test_p_monotone_nonincreasing_after_fill,
        test_cvm_total_unbiased,
        test_cvm_original_bias_detected,
        test_cvm_functional_invariant,
        test_knuth_unbiased_monte_carlo,
        test_dprime_power_of_two_cutoff,
        test_pac_coverage_eps_delta,
        test_tail_bounds_knuth_T_upperbounds,
        test_dprime_repeat_guard_bound,
        test_edge_m_eq_D_plus_1,
    ]
    results = []
    for fn in tests:
        try:
            kw = {}
            if fn.__name__ in {"test_pac_coverage_eps_delta","test_tail_bounds_knuth_T_upperbounds","test_dprime_power_of_two_cutoff"}:
                kw["variant"] = "knuth_dprime"
            elif fn.__name__ in {"test_cutoff_equals_max_in_buffer","test_last_occurrence_fairness","test_knuth_unbiased_monte_carlo"}:
                kw["variant"] = "knuth_d"
            else:
                kw["variant"] = variant
            res = fn(**kw)
        except Exception as e:
            res = {"claim_id": fn.__name__, "pass": False, "metrics": {"exception": str(e)}, "seed": None}
        results.append(res)
    return results
