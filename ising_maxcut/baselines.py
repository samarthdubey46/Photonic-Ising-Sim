"""Classical baselines to compare the CIM against."""

import numpy as np

from .graph import cut_value


def simulated_annealing(W, steps=20000, t_start=2.0, t_end=0.01,
                        seed=None, record=False):
    """Metropolis annealing for Max-Cut. Returns (best_spins, best_cut, history).

    A flip is accepted if it improves the cut, or with probability
    exp(gain / T) otherwise, while T cools geometrically.
    """
    n = len(W)
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=n)
    temps = np.geomspace(t_start, t_end, steps)

    cut = cut_value(W, s)
    best_cut, best_spins = cut, s.copy()
    history = np.empty(steps) if record else None

    for t, temp in enumerate(temps):
        i = rng.integers(n)
        gain = s[i] * (W[i] @ s)
        if gain > 0 or rng.random() < np.exp(gain / temp):
            s[i] = -s[i]
            cut += gain
            if cut > best_cut:
                best_cut, best_spins = cut, s.copy()
        if record:
            history[t] = best_cut

    return best_spins, best_cut, history


def random_baseline(W, samples=2000, seed=None):
    """Best cut over `samples` random partitions."""
    n = len(W)
    rng = np.random.default_rng(seed)
    best_cut, best_spins = -np.inf, None
    for _ in range(samples):
        s = rng.choice([-1.0, 1.0], size=n)
        cut = cut_value(W, s)
        if cut > best_cut:
            best_cut, best_spins = cut, s
    return best_spins, best_cut
