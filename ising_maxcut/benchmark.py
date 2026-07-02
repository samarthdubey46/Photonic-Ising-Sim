"""Benchmark harness: run a solver many times and score it like the papers do."""

import time
from dataclasses import dataclass

import numpy as np

from .cim import solve_cim
from .baselines import simulated_annealing
from .postprocess import greedy_local_search


@dataclass
class Result:
    name: str
    best_cut: float
    cuts: np.ndarray
    success_prob: float          # fraction of runs reaching the reference
    tts_99: float                # wall-time for a 99%-confidence hit
    seconds_per_run: float
    reference: float

    @property
    def pct_of_reference(self):
        return 100.0 * self.best_cut / self.reference


def _tts_99(success_prob, seconds_per_run):
    if success_prob >= 1.0:
        return seconds_per_run
    if success_prob <= 0.0:
        return float("inf")
    return seconds_per_run * np.log(0.01) / np.log(1 - success_prob)


def _score(name, cuts, reference, elapsed, runs, tol=1e-6):
    success_prob = float(np.mean(cuts >= reference - tol))
    per_run = elapsed / runs
    return Result(name, float(cuts.max()), cuts, success_prob,
                  _tts_99(success_prob, per_run), per_run, reference)


def benchmark_cim(W, reference, runs=50, postprocess=False, seed=0, **cim_kwargs):
    cuts = np.empty(runs)
    start = time.perf_counter()
    for r in range(runs):
        spins, cut, _ = solve_cim(W, seed=seed + r, **cim_kwargs)
        if postprocess:
            spins, cut = greedy_local_search(W, spins)
        cuts[r] = cut
    name = "CIM + local search" if postprocess else "CIM (raw)"
    return _score(name, cuts, reference, time.perf_counter() - start, runs)


def benchmark_sa(W, reference, runs=50, seed=0, **sa_kwargs):
    cuts = np.empty(runs)
    start = time.perf_counter()
    for r in range(runs):
        _, cut, _ = simulated_annealing(W, seed=seed + r, **sa_kwargs)
        cuts[r] = cut
    return _score("Simulated annealing", cuts, reference,
                  time.perf_counter() - start, runs)


def summarise(results):
    header = f"{'solver':<22}{'best':>8}{'%ref':>8}{'P_success':>11}{'TTS_99(s)':>12}{'s/run':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        tts = "inf" if r.tts_99 == float("inf") else f"{r.tts_99:.3f}"
        print(f"{r.name:<22}{r.best_cut:>8.0f}{r.pct_of_reference:>7.1f}%"
              f"{r.success_prob:>11.2f}{tts:>12}{r.seconds_per_run:>10.4f}")
