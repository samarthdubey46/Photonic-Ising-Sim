"""Plots for the benchmark: convergence traces and cut-value distributions."""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def plot_convergence(traces, labels, reference=None, title="", path=None):
    """traces: list of 1-D running-best-cut arrays (possibly different lengths)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tr, lab in zip(traces, labels):
        ax.plot(np.linspace(0, 1, len(tr)), tr, label=lab, lw=1.8)
    if reference is not None:
        ax.axhline(reference, ls="--", c="k", lw=1, label="reference (best known)")
    ax.set_xlabel("fraction of run")
    ax.set_ylabel("best cut so far")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_distributions(cut_sets, labels, reference=None, title="", path=None):
    """Histogram of final cut values for each solver, to show the shift the
    post-processor produces."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lo = min(c.min() for c in cut_sets)
    hi = max(c.max() for c in cut_sets)
    bins = np.linspace(lo - 1, hi + 1, 30)
    for cuts, lab in zip(cut_sets, labels):
        ax.hist(cuts, bins=bins, alpha=0.55, label=lab)
    if reference is not None:
        ax.axvline(reference, ls="--", c="k", lw=1, label="reference")
    ax.set_xlabel("cut value reached")
    ax.set_ylabel("number of runs")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
