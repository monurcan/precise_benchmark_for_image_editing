import random

import numpy as np


def generate_bimodal_sample(
    mean_1: float, std_1: float, mean_2: float, std_2: float, alpha_1: float
) -> float:
    """
    Generates a single sample from a bimodal distribution based on two normal distributions.

    Parameters:
    mean_1 (float): Mean of the first distribution.
    std_1 (float): Std deviation of the first distribution.
    mean_2 (float): Mean of the second distribution.
    std_2 (float): Std deviation of the second distribution.
    alpha_1 (float): Probability of selecting from the first distribution (0 <= alpha_1 <= 1).

    Returns:
    float: A single sample from the bimodal distribution.
    """
    # Randomly choose which distribution to sample from
    if np.random.rand() < alpha_1:
        # Sample from the first normal distribution
        return np.random.normal(mean_1, std_1)
    else:
        # Sample from the second normal distribution
        return np.random.normal(mean_2, std_2)


def random_sign():
    return 1 if random.random() < 0.5 else -1


if __name__ == "__main__":
    # Histogram of 1000 samples from a bimodal distribution
    samples = [
        max(0, generate_bimodal_sample(0.66, 0.15, 2.35, 0.6, 0.36))
        for _ in range(10000)
    ]
    import matplotlib.pyplot as plt

    plt.hist(samples, bins=100)
    plt.show()
