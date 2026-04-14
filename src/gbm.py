'''
gbm.py
Generate stock price paths and time steps for delta hedging using Geometric Brownian Motion
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class GBMConfig:
    '''Configuration for GBM simulation'''
    S0: float  # Initial stock price
    r: float  # Expected return (drift)
    sigma: float  # Volatility
    T: float  # Time horizon (in years)
    N: int  # Number of time steps
    M: int  # Number of paths to simulate
    seed: Optional[int] = 42 # Random seed for reproducibility


def generate_time_grid(T: float, N: int) -> np.ndarray:
    '''
    Generate time grid for GBM simulation
    T : Total time to maturity
    N : Number of time steps
    '''
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")
    if N <= 0:
        raise ValueError("Number of time steps N must be positive.")
    return np.linspace(0.0, T, N + 1)

def simulate_gbm_paths(
        S0: float, r: float, sigma: float, T: float, N: int, M: int, seed: Optional[int] = 42
) -> np.ndarray:
    '''
    Simulate GBM paths
    '''
    if S0 <= 0:
        raise ValueError("Initial stock price S0 must be positive.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")
    if M <= 0:
        raise ValueError("Number of paths M must be positive.")
    if N <= 0:
        raise ValueError("Number of time steps N must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")

    dt = T / N
    rng = np.random.default_rng(seed)

    # standard normal shocks : (M, N) shape
    Z = rng.standard_normal((M, N))

    # log-return increments
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    # cumulative log returns
    log_paths = np.cumsum(log_returns, axis=1)

    # prepend zeros for initial price
    log_paths = np.concatenate((np.zeros((M, 1)), log_paths), axis=1)

    # convert log paths to price paths
    paths = S0 * np.exp(log_paths)
    return paths

def theoretical_terminal_mean(S0: float, r: float, sigma: float, T: float) -> float:
    '''Calculate theoretical mean of terminal stock price'''
    return S0 * np.exp(r * T)

def theoretical_terminal_variance(S0: float, r: float, sigma: float, T: float) -> float:
    '''Calculate theoretical variance of terminal stock price'''
    return S0 ** 2 * np.exp(2 * r * T) * (np.exp(sigma ** 2 * T) - 1.0)

def validate_gbm_paths(
        paths: np.ndarray,
        S0: float,
        r: float,
        sigma: float,
        T: float) -> dict:
    '''Validate simulated GBM paths against theoretical properties '''

    if paths.ndim != 2:
        raise ValueError("Paths array must be 2-dimensional (M, N+1).")

    # Calculate empirical mean and variance of terminal stock prices
    terminal_prices = paths[:, -1]
    empirical_mean = np.mean(terminal_prices)
    empirical_variance = np.var(terminal_prices)

    # Calculate theoretical mean and variance
    theoretical_mean = theoretical_terminal_mean(S0, r, sigma, T)
    theoretical_variance = theoretical_terminal_variance(S0, r, sigma, T)

    # Return validation results
    return {
        "empirical_mean": empirical_mean,
        "theoretical_mean": theoretical_mean,
        "empirical_variance": empirical_variance,
        "theoretical_variance": theoretical_variance,
        "mean_error": abs(empirical_mean - theoretical_mean),
        "variance_error": abs(empirical_variance - theoretical_variance),
    }

def plot_sample_paths(
        paths: np.ndarray,
        T: float,
        N : int = 10,
        title: str = "Sample GBM Paths") -> None:
    '''Plot subset of simulated GBM paths'''

    if paths.ndim != 2:
        raise ValueError("Paths array must be 2-dimensional (M, N+1).")


    M, N_plus_1 = paths.shape
    if M == 0:
        raise ValueError("No paths to plot.")
    
    n_plot = min(N, M)
    time_grid = generate_time_grid(T, N_plus_1 - 1) 
    plt.figure(figsize=(10, 6))
    for i in range(n_plot):
        plt.plot(time_grid, paths[i, :], linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()

def demo()-> None:
    ''' 
    Run a simple demo 
    - simulate paths
    - validate terminal moments
    - plot sample paths
    '''

    cfg = GBMConfig(
        S0=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=252, # daily steps for 1 year 
        M=5000, # Monte Carlo paths
        seed=42
    )

    time_grid = generate_time_grid(cfg.T, cfg.N)
    paths = simulate_gbm_paths(cfg.S0, cfg.r, cfg.sigma, cfg.T, cfg.N, cfg.M, cfg.seed)
    summary = validate_gbm_paths(paths, cfg.S0, cfg.r, cfg.sigma, cfg.T)
    
    print("Time Grid:", len(time_grid), "steps from 0 to", cfg.T)
    print("Time interval dt:", time_grid[1] - time_grid[0])

    print("Validation Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    print("Plotting sample paths...")
    plot_sample_paths(paths, cfg.T, N=10, title="Sample GBM Paths")

if __name__ == "__main__":
    demo()


