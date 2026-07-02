"""Quickstart demo. Run:  python demo.py

Part A: a small graph with a known exact optimum (brute force), so success
        probability and time-to-solution are exact.
Part B: a larger graph scored as % of best-known, the way Gset benchmarking
        works, showing what the local-search output layer adds.
"""

import os

import numpy as np

from ising_maxcut import (
    random_graph, planted_bisection, brute_force_maxcut,
    solve_cim, simulated_annealing, greedy_local_search,
    cut_value, benchmark_cim, benchmark_sa, summarise,
)
from ising_maxcut.plotting import plot_convergence, plot_distributions

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = 60


def best_known(W, restarts=20, seed=0, **cim_kwargs):
    """A strong reference cut: best over several post-processed CIM runs."""
    best = 0.0
    for r in range(restarts):
        spins, _, _ = solve_cim(W, seed=seed + r, **cim_kwargs)
        _, cut = greedy_local_search(W, spins)
        best = max(best, cut)
    return best


def part_a():
    print("\n=== Part A: n=18, exact optimum known ===")
    W = random_graph(18, edge_prob=0.5, seed=3)
    _, optimum = brute_force_maxcut(W)
    print(f"exact max cut = {optimum:.0f}\n")

    results = [
        benchmark_cim(W, optimum, runs=RUNS, steps=1200, noise=0.05),
        benchmark_cim(W, optimum, runs=RUNS, steps=1200, noise=0.05, postprocess=True),
        benchmark_sa(W, optimum, runs=RUNS, steps=8000),
    ]
    summarise(results)
    plot_distributions(
        [results[0].cuts, results[1].cuts], ["CIM (raw)", "CIM + local search"],
        reference=optimum, title="n=18: the output layer shifts every run upward",
        path=os.path.join(HERE, "fig_distributions.png"),
    )


def part_b():
    print("\n=== Part B: n=140 planted graph, scored as % of best-known ===")
    W, planted = planted_bisection(140, p_within=0.1, p_across=0.7, seed=5)
    reference = max(best_known(W, steps=2500, noise=0.04), cut_value(W, planted))
    print(f"best-known cut (reference) = {reference:.0f}\n")

    results = [
        benchmark_cim(W, reference, runs=RUNS, steps=2500, noise=0.04),
        benchmark_cim(W, reference, runs=RUNS, steps=2500, noise=0.04, postprocess=True),
        benchmark_sa(W, reference, runs=RUNS, steps=40000),
    ]
    summarise(results)

    _, _, cim_trace = solve_cim(W, seed=0, steps=2500, noise=0.04, record=True)
    _, _, sa_trace = simulated_annealing(W, seed=0, steps=40000, record=True)
    plot_convergence(
        [cim_trace, sa_trace], ["CIM", "simulated annealing"], reference=reference,
        title="n=140: best cut over a single run",
        path=os.path.join(HERE, "fig_convergence.png"),
    )


if __name__ == "__main__":
    part_a()
    part_b()
    print(f"\nfigures written to {HERE}/")
