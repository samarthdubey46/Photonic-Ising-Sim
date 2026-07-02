"""Coherent-Ising-Machine simulator: integrate the loop's dynamics for Max-Cut."""

import numpy as np

from .graph import cut_value


def _spins(x):
    return np.where(x >= 0, 1.0, -1.0)


def _default_coupling(W, n):
    spectral_scale = np.abs(np.linalg.eigvalsh(W)).max()
    return 0.5 / spectral_scale if spectral_scale > 0 else 0.5 / np.sqrt(n)


def solve_cim(W, steps=1500, dt=0.05, coupling=None, noise=0.05,
              pump=(-2.0, 1.0), seed=None, record=False):
    """Run one CIM trajectory on W and return (best_spins, best_cut, history).

    Each soft spin x_i follows dx_i/dt = (p-1) x_i - x_i^3 - coupling (W x)_i,
    integrated with noise (Euler-Maruyama). The pump p ramps past threshold to
    anneal; readout is sign(x_i). We keep the best cut seen along the way.
    """
    n = len(W)
    rng = np.random.default_rng(seed)
    if coupling is None:
        coupling = _default_coupling(W, n)

    x = 0.01 * rng.standard_normal(n)
    pump_schedule = np.linspace(pump[0], pump[1], steps)
    noise_scale = np.sqrt(2 * noise * dt)

    best_spins = _spins(x)
    best_cut = cut_value(W, best_spins)
    history = np.empty(steps) if record else None

    for t, p in enumerate(pump_schedule):
        drift = (p - 1) * x - x ** 3 - coupling * (W @ x)
        x += dt * drift + noise_scale * rng.standard_normal(n)

        spins = _spins(x)
        cut = cut_value(W, spins)
        if cut > best_cut:
            best_cut, best_spins = cut, spins
        if record:
            history[t] = best_cut

    return best_spins, best_cut, history
