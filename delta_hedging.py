"""
Part 3: Delta hedging simulation with periodic rebalancing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import OptionConfig, black_scholes_delta, black_scholes_price


@dataclass
class HedgeResult:
    option_premium: float
    final_pnl: np.ndarray
    hedging_error: np.ndarray
    stock_positions: np.ndarray
    cash_positions: np.ndarray
    portfolio_values: np.ndarray
    option_values: np.ndarray
    hedge_deltas: np.ndarray
    time_grid: np.ndarray
    summary: dict


def option_payoff(stock_prices: np.ndarray, strike: float, option_type: str = "call") -> np.ndarray:
    if option_type == "call":
        return np.maximum(stock_prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - stock_prices, 0.0)
    raise ValueError("option_type must be 'call' or 'put'.")


def run_delta_hedge(
    stock_paths: np.ndarray,
    r: float,
    sigma: float,
    option: OptionConfig,
    rebalance_every: int = 1,
) -> HedgeResult:
    if stock_paths.ndim != 2:
        raise ValueError("stock_paths must have shape (M, N+1).")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be a positive integer.")

    num_paths, num_points = stock_paths.shape
    steps = num_points - 1
    dt = option.maturity / steps
    time_grid = np.linspace(0.0, option.maturity, num_points)

    stock_positions = np.zeros((num_paths, num_points))
    cash_positions = np.zeros((num_paths, num_points))
    portfolio_values = np.zeros((num_paths, num_points))
    option_values = np.zeros((num_paths, num_points))
    hedge_deltas = np.zeros((num_paths, num_points))

    initial_tau = option.maturity
    initial_prices = stock_paths[:, 0]
    initial_option_values = black_scholes_price(
        initial_prices,
        option.strike,
        r,
        sigma,
        initial_tau,
        option.option_type,
    )
    initial_deltas = black_scholes_delta(
        initial_prices,
        option.strike,
        r,
        sigma,
        initial_tau,
        option.option_type,
    )

    # We short the option, receive the premium, and hold delta shares.
    stock_positions[:, 0] = initial_deltas
    cash_positions[:, 0] = initial_option_values - initial_deltas * initial_prices
    option_values[:, 0] = initial_option_values
    hedge_deltas[:, 0] = initial_deltas
    portfolio_values[:, 0] = cash_positions[:, 0] + stock_positions[:, 0] * initial_prices - option_values[:, 0]

    for step in range(1, num_points):
        current_prices = stock_paths[:, step]
        remaining_tau = max(option.maturity - time_grid[step], 0.0)

        cash_positions[:, step] = cash_positions[:, step - 1] * np.exp(r * dt)
        stock_positions[:, step] = stock_positions[:, step - 1]

        if step < num_points - 1 and step % rebalance_every == 0:
            new_delta = black_scholes_delta(
                current_prices,
                option.strike,
                r,
                sigma,
                remaining_tau,
                option.option_type,
            )
            delta_change = new_delta - stock_positions[:, step]
            cash_positions[:, step] -= delta_change * current_prices
            stock_positions[:, step] = new_delta
            hedge_deltas[:, step] = new_delta
        else:
            hedge_deltas[:, step] = stock_positions[:, step]

        option_values[:, step] = black_scholes_price(
            current_prices,
            option.strike,
            r,
            sigma,
            remaining_tau,
            option.option_type,
        )
        portfolio_values[:, step] = (
            cash_positions[:, step]
            + stock_positions[:, step] * current_prices
            - option_values[:, step]
        )

    payoff = option_payoff(stock_paths[:, -1], option.strike, option.option_type)
    final_pnl = cash_positions[:, -1] + stock_positions[:, -1] * stock_paths[:, -1] - payoff
    hedging_error = final_pnl

    summary = {
        "mean_hedging_error": float(np.mean(hedging_error)),
        "std_hedging_error": float(np.std(hedging_error, ddof=1)),
        "min_hedging_error": float(np.min(hedging_error)),
        "max_hedging_error": float(np.max(hedging_error)),
        "rmse_hedging_error": float(np.sqrt(np.mean(hedging_error**2))),
        "mean_final_pnl": float(np.mean(final_pnl)),
        "probability_loss": float(np.mean(final_pnl < 0.0)),
    }

    return HedgeResult(
        option_premium=float(initial_option_values[0]),
        final_pnl=final_pnl,
        hedging_error=hedging_error,
        stock_positions=stock_positions,
        cash_positions=cash_positions,
        portfolio_values=portfolio_values,
        option_values=option_values,
        hedge_deltas=hedge_deltas,
        time_grid=time_grid,
        summary=summary,
    )


def run_simulation(
    stock_paths: np.ndarray,
    r: float,
    sigma: float,
    option: OptionConfig,
    rebalance_every: int = 1,
) -> np.ndarray:
    result = run_delta_hedge(
        stock_paths=stock_paths,
        r=r,
        sigma=sigma,
        option=option,
        rebalance_every=rebalance_every,
    )
    return result.hedging_error
