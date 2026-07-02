"""Output layer: greedy local search that cleans up a solver's raw spins."""

import numpy as np

from .graph import cut_value, flip_gains


def greedy_local_search(W, spins, max_sweeps=10000):
    """Repeatedly flip the most-improving spin until none helps (local optimum).
    Returns (spins, cut)."""
    s = np.array(spins, dtype=float)
    for _ in range(max_sweeps):
        gains = flip_gains(W, s)
        i = int(gains.argmax())
        if gains[i] <= 1e-9:
            break
        s[i] = -s[i]
    return s, cut_value(W, s)
