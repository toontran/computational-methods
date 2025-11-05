
import numpy as np

def perm_with_repeats(D:int, m:int, seed:int, repeat_factor:float=1.0):
    """
    Universe of D distinct items labeled 0..D-1. Stream length m.
    repeat_factor >= 1.0 controls tendency to repeat recently seen items.
    """
    rng = np.random.RandomState(seed)
    items = np.arange(D)
    stream = []
    last = None
    for t in range(m):
        if (last is not None) and (rng.rand() < (1.0 - 1.0/repeat_factor)):
            stream.append(last)
        else:
            x = int(items[rng.randint(0, D)])
            stream.append(x)
            last = x
    return stream, D

def zipf_stream(D:int, m:int, seed:int, alpha:float=1.0):
    """
    Keys 0..D-1 sampled i.i.d. from Zipf(alpha) truncated to D.
    """
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, D+1)
    weights = ranks ** (-alpha)
    probs = weights / weights.sum()
    stream = list(rng.choice(np.arange(D), size=m, p=probs))
    return stream, D

def adversarial_repeats(D:int, m:int, seed:int, block:int=10):
    """
    Present each key in long repeated blocks to stress last-appearance logic.
    """
    rng = np.random.RandomState(seed)
    order = list(rng.permutation(D))
    stream = []
    i = 0
    while len(stream) < m:
        key = order[i % D]
        stream.extend([key]*block)
        i += 1
    return stream[:m], D

def edge_m_eq_D_plus_1(D:int, seed:int):
    """
    Stream of length m=D+1: present all D keys once, then repeat key 0 once.
    """
    stream = list(range(D)) + [0]
    return stream, D

def true_F0(stream):
    return len(set(stream))
