""" 
Simulation module for PV-cell.
Based on the real signals that looks somewhat like a skewed normal distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


def simulate_pv_cell(
    samples_per_hour=6,
    max_power=20,
    sunrise=8,
    sunset=16,
    days=1,
    add_noise=True,
    logpath="",
    plot=True,
):
    """Simulation of PV-cell

    Uses a skew normal distribution, which approximates a PV-cell.
    All values outside sunset-sunrise is set to zero

    params:
        resolution  [min]: Sampling time. Defaults to 10min -> 6 samples per hour
        max_power   [kW]: Scaling-factor (dependent on irridation/weather/season)
        sunrise     [h]: Time where PV production starts
        sunset      [h]: Time where PV production stops
        add_noise   [bool]: Add gaussion noise to measurements
        plot        [bool]: Plots the powerproduction during the day

    returns:
        P_pv        [kW]: array with 24_samples_per_hour of pv_values
    """

    HOURS = 24
    ACTIVE_HOURS = sunset - sunrise
    NUM_ACTIVE_DATAPOINTS = samples_per_hour * ACTIVE_HOURS
    SKEWING_FACTOR = -3

    x = np.linspace(
        skewnorm.ppf(0.1, SKEWING_FACTOR),
        skewnorm.ppf(0.99, SKEWING_FACTOR),
        NUM_ACTIVE_DATAPOINTS,
    )

    if add_noise:
        n = np.random.normal(0, 0.05, NUM_ACTIVE_DATAPOINTS)
        n[::2] = 0
    else:
        n = np.zeros(NUM_ACTIVE_DATAPOINTS)

    skewnorm_ = skewnorm.pdf(x, SKEWING_FACTOR, loc=0, scale=0.5)

    pv_values = max_power * np.clip((skewnorm_ + n), 0, np.inf)

    t_pre_sunset = np.zeros(sunrise * samples_per_hour)
    t_post_sunset = np.zeros((HOURS - sunset) * samples_per_hour)
    t = np.linspace(0, days * HOURS, num=days * HOURS * samples_per_hour)

    P_pv_final = []
    for _ in range(0, days):
        weather = np.random.randint(4, 5)
        if weather < 4:  # Rain or fully cloudy
            pv_values = max_power * 0.2 * np.clip((skewnorm_ + n), 0, np.inf)
        elif weather > 6:  # Partially cloudy
            pv_values = max_power * np.clip((skewnorm_ + n), 0, np.inf)
            pv_values = np.multiply(
                pv_values, np.random.rand(ACTIVE_HOURS * samples_per_hour)
            )
        else:  # Sunny day
            pv_values = max_power * np.clip((skewnorm_ + n), 0, np.inf)
        P_pv = np.concatenate((t_pre_sunset, pv_values, t_post_sunset), axis=None)
        P_pv_final = np.concatenate((P_pv_final, P_pv), axis=None)

    if plot:
        # Plot
        plt.figure()
        title = "Simulated PV-Cell"
        plt.plot(t, P_pv_final)

        plt.xlabel("Time [h]")
        plt.ylabel("kW")
        plt.title(title)
        if logpath:
            plt.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")

    print(
        "Created PV-cell simulation with add_noise = {}. Max_value = {}".format(
            add_noise, np.max(P_pv_final)
        )
    )

    return P_pv_final


if __name__ == "__main__":
    simulate_pv_cell(days=1)
    plt.show()
