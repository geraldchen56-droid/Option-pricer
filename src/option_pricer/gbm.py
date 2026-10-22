"""Simulating the price of a stock at a future date."""

import math

import matplotlib
matplotlib.use("Agg")  # save to a file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np


def simulate_terminal_prices(S0, r, sigma, T, n_sims, Z=None):
    """Simulate n_sims possible prices of the stock at time T.

    S_T = S0 * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z)

    Z is a standard normal random number: average 0, standard deviation 1.
    One Z per simulation gives one possible future price.

    Pass Z to supply your own random numbers instead of drawing fresh ones.
    Reusing the same Z across several parameter settings means the differences
    between the results come from the parameters, not from new randomness.
    """
    if Z is None:
        Z = np.random.standard_normal(n_sims)

    drift = (r - 0.5 * sigma**2) * T          # the predictable part
    shock = sigma * math.sqrt(T) * Z          # the random part

    return S0 * np.exp(drift + shock)


# Only runs when this file is executed directly, not when another module
# imports simulate_terminal_prices from it.
if __name__ == "__main__":
    S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    prices = simulate_terminal_prices(S0, r, sigma, T, 100_000)

    sample_mean = prices.mean()
    theoretical_mean = S0 * math.exp(r * T)

    print(f"sample mean of S_T:  {sample_mean:.4f}")
    print(f"S0 * e^(rT):         {theoretical_mean:.4f}")

    plt.hist(prices, bins=100)
    plt.xlabel("terminal price S_T")
    plt.ylabel("number of simulations")
    plt.title("100,000 simulated terminal prices")
    plt.savefig("gbm_histogram.png", dpi=120)
    print("histogram saved to gbm_histogram.png")
