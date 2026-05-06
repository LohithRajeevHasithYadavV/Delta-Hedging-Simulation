from __future__ import annotations

import numpy as np

from .bs_model import bs_call_price, bs_delta


def delta_hedge_single_path(S_path: np.ndarray, K: float, T: float, r: float, sigma: float, dt: float) -> dict:
    N = len(S_path) - 1
    time = np.linspace(0.0, T, N + 1)

    option_values = np.zeros(N + 1)
    deltas = np.zeros(N + 1)
    stock_positions = np.zeros(N + 1)
    cash_positions = np.zeros(N + 1)
    portfolio_values = np.zeros(N + 1)

    option_values[0] = float(bs_call_price(S_path[0], K, T, r, sigma))
    deltas[0] = float(bs_delta(S_path[0], K, T, r, sigma))
    stock_positions[0] = deltas[0]
    cash_positions[0] = option_values[0] - stock_positions[0] * S_path[0]
    portfolio_values[0] = cash_positions[0] + stock_positions[0] * S_path[0] - option_values[0]

    for t in range(1, N + 1):
        tau = max(T - t * dt, 0.0)
        cash_positions[t] = cash_positions[t - 1] * np.exp(r * dt)
        stock_positions[t] = stock_positions[t - 1]

        if t < N:
            new_delta = float(bs_delta(S_path[t], K, tau, r, sigma))
            delta_change = new_delta - stock_positions[t]
            cash_positions[t] -= delta_change * S_path[t]
            stock_positions[t] = new_delta
            deltas[t] = new_delta
        else:
            deltas[t] = stock_positions[t]

        option_values[t] = float(bs_call_price(S_path[t], K, tau, r, sigma))
        portfolio_values[t] = cash_positions[t] + stock_positions[t] * S_path[t] - option_values[t]

    payoff = max(S_path[-1] - K, 0.0)
    final_value = cash_positions[-1] + stock_positions[-1] * S_path[-1] - payoff

    return {
        "time": time,
        "option_values": option_values,
        "deltas": deltas,
        "stock_positions": stock_positions,
        "cash_positions": cash_positions,
        "portfolio_values": portfolio_values,
        "payoff": payoff,
        "final_value": final_value,
        "error": final_value,
    }


def compute_final_pnl(S_path: np.ndarray, K: float, T: float, r: float, sigma: float, dt: float) -> tuple[float, float, float]:
    hedge = delta_hedge_single_path(S_path, K, T, r, sigma, dt)
    return hedge["final_value"], hedge["payoff"], hedge["error"]


def run_delta_hedge(paths: np.ndarray, K: float, T: float, r: float, sigma: float, rebalance_every: int = 1) -> dict:
    if paths.ndim != 2:
        raise ValueError("paths must have shape (M, N + 1).")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive.")

    M, N_plus_1 = paths.shape
    N = N_plus_1 - 1
    dt = T / N
    time_grid = np.linspace(0.0, T, N_plus_1)

    stock_positions = np.zeros((M, N_plus_1))
    cash_positions = np.zeros((M, N_plus_1))
    option_values = np.zeros((M, N_plus_1))
    portfolio_values = np.zeros((M, N_plus_1))
    delta_history = np.zeros((M, N_plus_1))

    option_values[:, 0] = bs_call_price(paths[:, 0], K, T, r, sigma)
    delta_history[:, 0] = bs_delta(paths[:, 0], K, T, r, sigma)
    stock_positions[:, 0] = delta_history[:, 0]
    cash_positions[:, 0] = option_values[:, 0] - stock_positions[:, 0] * paths[:, 0]
    portfolio_values[:, 0] = cash_positions[:, 0] + stock_positions[:, 0] * paths[:, 0] - option_values[:, 0]

    for t in range(1, N_plus_1):
        tau = max(T - t * dt, 0.0)
        cash_positions[:, t] = cash_positions[:, t - 1] * np.exp(r * dt)
        stock_positions[:, t] = stock_positions[:, t - 1]

        if t < N and t % rebalance_every == 0:
            new_delta = bs_delta(paths[:, t], K, tau, r, sigma)
            delta_change = new_delta - stock_positions[:, t]
            cash_positions[:, t] -= delta_change * paths[:, t]
            stock_positions[:, t] = new_delta
            delta_history[:, t] = new_delta
        else:
            delta_history[:, t] = stock_positions[:, t]

        option_values[:, t] = bs_call_price(paths[:, t], K, tau, r, sigma)
        portfolio_values[:, t] = cash_positions[:, t] + stock_positions[:, t] * paths[:, t] - option_values[:, t]

    payoff = np.maximum(paths[:, -1] - K, 0.0)
    final_values = cash_positions[:, -1] + stock_positions[:, -1] * paths[:, -1] - payoff
    errors = final_values.copy()

    return {
        "time_grid": time_grid,
        "portfolio_values": portfolio_values,
        "cash_positions": cash_positions,
        "stock_positions": stock_positions,
        "option_values": option_values,
        "delta_history": delta_history,
        "final_values": final_values,
        "errors": errors,
        "payoffs": payoff,
        "summary": {
            "mean_hedging_error": float(np.mean(errors)),
            "std_hedging_error": float(np.std(errors, ddof=1)),
            "min_hedging_error": float(np.min(errors)),
            "max_hedging_error": float(np.max(errors)),
            "rmse_hedging_error": float(np.sqrt(np.mean(errors ** 2))),
            "mean_final_pnl": float(np.mean(final_values)),
            "probability_of_loss": float(np.mean(final_values < 0.0)),
        },
    }
