"""Resampling statistics.

- ``block_bootstrap_mean``: confidence interval for mean trade return. Uses moving blocks so
  short-range dependence between consecutive trades is not destroyed. Answers: how uncertain is the
  average edge given this sample?
- ``shuffle_drawdown``: Monte Carlo over trade *order*. Assumes trade outcomes are exchangeable
  (stated assumption; violated under strong serial dependence). Answers: how bad could drawdown
  have been with the same trades in a different order? It says nothing about the mean.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    prob_mean_le_zero: float
    n: int
    resamples: int
    block: int
    seed: int


def block_bootstrap_mean(values: list[float], *, resamples: int = 2000, block: int = 3, seed: int = 0,
                         alpha: float = 0.05) -> BootstrapCI | None:
    n = len(values)
    if n < 10:
        return None
    block = max(1, min(block, n))
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(0, n - block + 1)
            sample.extend(values[start:start + block])
        sample = sample[:n]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return BootstrapCI(sum(values) / n, lo, hi, sum(1 for m in means if m <= 0) / resamples, n, resamples, block, seed)


def shuffle_drawdown(pnls: list[float], initial_equity: float, *, runs: int = 2000, seed: int = 0) -> dict | None:
    if len(pnls) < 10:
        return None
    rng = random.Random(seed)
    dds = []
    for _ in range(runs):
        order = pnls[:]
        rng.shuffle(order)
        eq = peak = initial_equity
        worst = 0.0
        for p in order:
            eq += p
            peak = max(peak, eq)
            worst = max(worst, (peak - eq) / peak if peak > 0 else 1.0)
        dds.append(worst)
    dds.sort()
    return {"median": dds[runs // 2], "p95": dds[int(0.95 * runs)], "max": dds[-1], "runs": runs, "seed": seed,
            "assumption": "trade outcomes exchangeable"}
