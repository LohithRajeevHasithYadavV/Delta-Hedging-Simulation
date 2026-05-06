from __future__ import annotations

from math import erf, sqrt

import numpy as np


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    erf_values = np.vectorize(erf)(values / sqrt(2.0))
    return 0.5 * (1.0 + erf_values)


def d1(S: np.ndarray | float, K: float, T: np.ndarray | float, r: float, sigma: float) -> np.ndarray:
    S_values = np.asarray(S, dtype=float)
    T_values = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return (np.log(S_values / K) + (r + 0.5 * sigma ** 2) * T_values) / (sigma * np.sqrt(T_values))


def d2(S: np.ndarray | float, K: float, T: np.ndarray | float, r: float, sigma: float) -> np.ndarray:
    T_values = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return d1(S, K, T_values, r, sigma) - sigma * np.sqrt(T_values)


def bs_call_price(S: np.ndarray | float, K: float, T: np.ndarray | float, r: float, sigma: float) -> np.ndarray:
    S_values = np.asarray(S, dtype=float)
    T_values = np.asarray(T, dtype=float)
    intrinsic_value = np.maximum(S_values - K, 0.0)

    D1 = d1(S_values, K, T_values, r, sigma)
    D2 = d2(S_values, K, T_values, r, sigma)
    call_price = S_values * normal_cdf(D1) - K * np.exp(-r * np.maximum(T_values, 0.0)) * normal_cdf(D2)
    return np.where(T_values <= 0, intrinsic_value, call_price)


def bs_delta(S: np.ndarray | float, K: float, T: np.ndarray | float, r: float, sigma: float) -> np.ndarray:
    S_values = np.asarray(S, dtype=float)
    T_values = np.asarray(T, dtype=float)
    D1 = d1(S_values, K, T_values, r, sigma)
    delta = normal_cdf(D1)
    return np.where(T_values <= 0, np.where(S_values > K, 1.0, 0.0), delta)


def validate_black_scholes(K: float, T: float, r: float, sigma: float) -> dict:
    stock_grid = np.linspace(50.0, 150.0, 101)
    prices = bs_call_price(stock_grid, K, T, r, sigma)
    deltas = bs_delta(stock_grid, K, T, r, sigma)

    return {
        "initial_call_price": float(bs_call_price(100.0, K, T, r, sigma)),
        "initial_delta": float(bs_delta(100.0, K, T, r, sigma)),
        "min_price": float(np.min(prices)),
        "max_price": float(np.max(prices)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "price_increases_with_stock": bool(np.all(np.diff(prices) >= -1e-10)),
        "delta_between_zero_and_one": bool(np.all((deltas >= -1e-10) & (deltas <= 1.0 + 1e-10))),
    }
