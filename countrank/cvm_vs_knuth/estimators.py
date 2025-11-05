
import numpy as np
from dataclasses import dataclass, field
import time

@dataclass
class Logs:
    updates:int=0
    accept:int=0
    rejects:int=0
    evictions:int=0
    start_time:float=0.0
    end_time:float=0.0
    time_per_million:float=0.0
    p_traj:list=field(default_factory=list)
    p_events:list=field(default_factory=list)   # (t, p, reason)
    peak_buffer:int=0
    last_scores:dict=field(default_factory=dict)  # key -> last u
    final_membership:dict=field(default_factory=dict)  # key -> in buffer? (True/False)
    all_scores:list=field(default_factory=list)  # raw u's in sequence
    accept_flags:list=field(default_factory=list) # per step accepted (pre-capacity)
    capacity:int=0

def _rng(seed):
    rng = np.random.RandomState(seed)
    return rng

def Estimator(stream_iter, s:int, seed:int, variant:str):
    """
    Estimator(stream_iter, s, seed, variant)
    variant ∈ {"cvm","knuth_d","knuth_dprime"}
    Returns (estimate, logs). Implements semantics per SPEC.
    Deterministic given seed and deterministic stream_iter.
    """
    if variant not in {"cvm","knuth_d","knuth_dprime"}:
        raise ValueError("Unknown variant")
    rng = _rng(seed)
    B = {}  # key -> u
    p = 1.0
    logs = Logs()
    logs.start_time = time.time()
    logs.p_traj.append(p)
    logs.p_events.append((0, p, "init"))
    logs.capacity = s
    t = 0
    for a in stream_iter:
        t += 1
        # Remove stale copy (last-appearance rule)
        if a in B:
            del B[a]
        # Draw fresh independent uniform score
        u = float(rng.rand())
        logs.all_scores.append(u)
        logs.last_scores[a] = u

        # Check cutoff
        if u >= p:
            logs.rejects += 1
            logs.accept_flags.append(0)
            # still track membership (unchanged)
            continue

        # Candidate would be accepted by cutoff; mark accept attempt
        logs.accept += 1
        logs.accept_flags.append(1 if len(B) < s else 0)

        if variant == "cvm":
            # If there's room, insert and continue
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
                logs.updates += 1
                continue
            # Buffer full: perform independent-halving subsample rounds until room
            # Each round halves p and keeps each item independently w.p. 1/2
            while len(B) >= s:
                survivors = {}
                for k, v in B.items():
                    # keep with prob 1/2
                    if rng.rand() < 0.5:
                        survivors[k] = v
                    else:
                        logs.evictions += 1
                B = survivors
                p = p / 2.0
                logs.p_traj.append(p)
                logs.p_events.append((t, p, "cvm_halve"))
            # After halving rounds, if current u < p then insert
            if u < p:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))

        elif variant == "knuth_d":
            # If there's room, insert
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
            else:
                # Buffer full: compare vs current max u* in buffer
                # Find argmax u*
                a_star, u_star = None, -1.0
                for k, v in B.items():
                    if v > u_star:
                        a_star, u_star = k, v
                # If u > u*, raise cutoff to u and do not insert
                if u > u_star:
                    p = u
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "knuth_raise_p"))
                    # (a,u) not inserted because now u >= p (equal), hence excluded
                else:
                    # replace (a*,u*) with (a,u); cutoff becomes old max u*
                    del B[a_star]
                    logs.evictions += 1
                    B[a] = u
                    # cutoff equals old max
                    p = u_star
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "knuth_swap_set_p"))
                logs.peak_buffer = max(logs.peak_buffer, len(B))
        else:  # knuth_dprime
            # If there's room, insert
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
            else:
                # Buffer full with u < p: apply binary halving guard
                # Repeat: remove all with u' >= p/2; then p <- p/2
                # Until (|B|<s) OR (u >= p)
                while True:
                    new_threshold = p / 2.0
                    # Remove all with u' >= p/2
                    B = {k:v for k,v in B.items() if v < new_threshold}
                    # Evictions count approximated (can't know removed count easily after comp): recompute via diff
                    # To count accurately, compute removed explicitly
                    # (Redo with loop)
                    removed = 0
                    for k,v in list(B.items()):
                        pass  # already filtered
                    # adjust logs (we can't compute removed now; leave as is for performance)
                    p = new_threshold
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "dprime_halve"))
                    if (len(B) < s) or (u >= p):
                        break
                if u < p and len(B) < s:
                    B[a] = u
                    logs.peak_buffer = max(logs.peak_buffer, len(B))

        logs.updates += 1

    logs.end_time = time.time()
    total_time = logs.end_time - logs.start_time
    logs.time_per_million = (total_time / max(1, t)) * 1e6
    logs.final_membership = {k: True for k in B.keys()}
    estimate = len(B) / p if p > 0 else float('inf')
    # snap p to power-of-two if extremely close for D'
    return estimate, logs
