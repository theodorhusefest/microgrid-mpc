""" 
Simulation module for PV-load. 
Based on the real signals that looks somewhat like a skewed normal distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


def simulate_p_load(
    samples_per_hour=6,
    max_power=100,
    min_power=20,
    days=1,
    add_noise=True,
    logpath="",
    plot=True,
):
    """Simulation of PV-cell

    Uses a skew normal distribution, which approximates the load.
    All values outside sunset-sunrise is set to zero

    params:
        samples_per_hour  [min]: Sampling time. Defaults to 10min -> 6 samples per hour
        max_power   [kW]: Scaling-factor (dependent on irridation/weather/season)
        min_power   [kW]: Lifts the distribution to a minumum level
        add_noise   [bool]: Add gaussion noise to measurements
        plot        [bool]: Plots the power-load throughout the day

    returns:
        P_L        [kW]: array with 24_samples_per_hour of p_load
    """

    HOURS = 24
    SKEWING_FACTOR = 4
    NUM_ACTIVE_DATAPOINTS = samples_per_hour * HOURS

    x = np.linspace(
        skewnorm.ppf(0.1, SKEWING_FACTOR),
        skewnorm.ppf(0.9999, SKEWING_FACTOR),
        NUM_ACTIVE_DATAPOINTS,
    )

    if add_noise:
        n = np.random.normal(0, 0.05, NUM_ACTIVE_DATAPOINTS) * 0.3 * max_power
    else:
        n = np.zeros(NUM_ACTIVE_DATAPOINTS)

    skewnorm_ = skewnorm.pdf(x, SKEWING_FACTOR, loc=1, scale=1.5)

    P_L = max_power * skewnorm_ + min_power + n
    t = np.linspace(0, days * HOURS, num=days * HOURS * samples_per_hour)

    P_L = np.concatenate(([P_L for _ in range(days)]), axis=None)
    if plot:
        # Plot
        plt.figure()
        title = "P_load simulation"
        plt.plot(t, P_L)
        plt.xlabel("Time [h]")
        plt.ylabel("P_L [kW]")
        plt.title(title)
        plt.savefig("{}-{}".format(logpath, title))

    return P_L


if __name__ == "__main__":
    simulate_p_load(days=3)
    plt.show()