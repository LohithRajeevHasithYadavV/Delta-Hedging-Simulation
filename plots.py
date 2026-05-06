from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .bs_model import bs_delta


def plot_single_path(S_path: np.ndarray, hedge_data: dict, T: float, output_path: str) -> None:
    time = hedge_data["time"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(time, S_path, color="navy", linewidth=2)
    axes[0].set_ylabel("Stock Price")
    axes[0].set_title("Single Path Delta Hedging")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, hedge_data["deltas"], color="darkorange", linewidth=2)
    axes[1].set_ylabel("Delta")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, hedge_data["portfolio_values"], color="darkgreen", linewidth=2)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Portfolio Value")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(errors: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(errors), color="red", linestyle="--", linewidth=2)
    plt.title("Distribution of Hedging Error")
    plt.xlabel("Hedging Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_portfolio_paths(time_grid: np.ndarray, portfolio_values: np.ndarray, output_path: str, num_paths: int = 20) -> None:
    plt.figure(figsize=(10, 6))
    for i in range(min(num_paths, portfolio_values.shape[0])):
        plt.plot(time_grid, portfolio_values[i], linewidth=1.0, alpha=0.8)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Portfolio Value Through Time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_average_hedging_error(time_grid: np.ndarray, portfolio_values: np.ndarray, output_path: str) -> None:
    average_error = np.mean(portfolio_values, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, average_error, color="purple", linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Average Hedging Error Through Time")
    plt.xlabel("Time")
    plt.ylabel("Average Hedging Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_frequency_vs_error(results: dict, output_path: str) -> None:
    labels = list(results.keys())
    stds = [results[key]["std"] for key in labels]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, stds, marker="o", linewidth=2, color="maroon")
    plt.title("Rebalancing Frequency and Hedging Error")
    plt.xlabel("Rebalancing Frequency")
    plt.ylabel("Standard Deviation of Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_volatility_vs_error(results: dict, output_path: str) -> None:
    labels = list(results.keys())
    stds = [results[key]["std"] for key in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, stds, color=["#6baed6", "#fb6a4a"])
    plt.title("Volatility and Hedging Error")
    plt.xlabel("Scenario")
    plt.ylabel("Standard Deviation of Error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_moneyness_vs_error(results: dict, output_path: str) -> None:
    labels = list(results.keys())
    stds = [results[key]["std"] for key in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, stds, color=["#74c476", "#9e9ac8", "#fd8d3c"])
    plt.title("Moneyness and Hedging Error")
    plt.xlabel("Moneyness")
    plt.ylabel("Standard Deviation of Error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
