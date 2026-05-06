from __future__ import annotations

from .gbm import simulate_gbm_paths
from .hedging import run_delta_hedge


def run_simulation(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    M: int,
    seed: int = 42,
    rebalance_every: int = 1,
) -> tuple:
    paths = simulate_gbm_paths(S0, r, sigma, T, N, M, seed)
    hedge_data = run_delta_hedge(paths, K, T, r, sigma, rebalance_every=rebalance_every)
    return paths, hedge_data
