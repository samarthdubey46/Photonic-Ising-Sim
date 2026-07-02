"""Max-Cut objective and problem instances.

A problem is a weighted adjacency matrix W (symmetric, zero diagonal).
A solution is a spin vector s in {-1, +1}: s_i is which side vertex i is on.
Maximising cut_value(W, s) is equivalent to minimising (1/2) s^T W s.
"""

import numpy as np


def cut_value(W, s):
    """Total weight of edges crossing the partition defined by s."""
    return 0.25 * (W.sum() - s @ W @ s)


def ising_energy(W, s):
    """Ising energy; its minimum is the maximum cut."""
    return 0.5 * (s @ W @ s)


def flip_gains(W, s):
    """gains[i] = change in cut value if spin i is flipped, = s_i (W s)_i."""
    return s * (W @ s)


def random_graph(n, edge_prob=0.5, weighted=False, seed=None):
    """Erdos-Renyi graph, unit weights or integer weights in 1..10."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((n, n)) < edge_prob, k=1).astype(float)
    if weighted:
        upper *= rng.integers(1, 11, size=(n, n))
    W = upper + upper.T
    return W


def planted_bisection(n, p_within=0.1, p_across=0.7, seed=None):
    """Two blocks, dense across / sparse within, so the block split is a
    strong known cut. Returns (W, planted_spins)."""
    rng = np.random.default_rng(seed)
    side = np.array([0] * (n // 2) + [1] * (n - n // 2))
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_across if side[i] != side[j] else p_within
            if rng.random() < p:
                W[i, j] = W[j, i] = 1.0
    planted = np.where(side == 0, 1.0, -1.0)
    return W, planted


def load_gset(path):
    """Read a Gset-format file ('n m' header, then 'i j w' lines, 1-indexed)."""
    with open(path) as f:
        n, m = map(int, f.readline().split())
        W = np.zeros((n, n))
        for _ in range(m):
            i, j, w = f.readline().split()
            i, j = int(i) - 1, int(j) - 1
            W[i, j] = W[j, i] = float(w)
    return W


def brute_force_maxcut(W):
    """Exact Max-Cut by enumeration (n <= ~22). Spin 0 is fixed to +1."""
    n = len(W)
    best_cut, best_spins = -np.inf, None
    for bits in range(2 ** (n - 1)):
        s = np.ones(n)
        for k in range(n - 1):
            if (bits >> k) & 1:
                s[k + 1] = -1.0
        cut = cut_value(W, s)
        if cut > best_cut:
            best_cut, best_spins = cut, s.copy()
    return best_spins, best_cut
