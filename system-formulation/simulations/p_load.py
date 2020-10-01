""" 
Simulation module for PV-cell 
Based on the real signals that looks somewhat like a sinus with max-value around 75 kW
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def simulate_p_load(
    resulution= 10,
    max_power= 100,
    min_power= 20,
    days= 1,
    add_noise = True,
    plot = True):
    """ Simulation of PV-cell
    
    Uses a skew normal distribution, which approximates the load.
    All values outside sunset-sunrise is set to zero

    params:
        resolution  [min]: Sampling time. Defaults to 10min -> 6 samples per hour
        max_power   [kW]: Scaling-factor (dependent on irridation/weather/season)
        min_power   [kW]: Lifts the distribution to a minumum level
        add_noise   [bool]: Add gaussion noise to measurements
        plot        [bool]: Plots the power-load throughout the day

    returns:
        P_L        [kW]: array with 24_samples_per_hour of p_load
    """

    HOURS = 24
    SKEWING_FACTOR = 4
    SAMPLES_PER_HOUR= int(60/resulution)
    NUM_ACTIVE_DATAPOINTS= SAMPLES_PER_HOUR*HOURS

    x = np.linspace(skewnorm.ppf(0.1, SKEWING_FACTOR),
                skewnorm.ppf(0.9999, SKEWING_FACTOR), NUM_ACTIVE_DATAPOINTS)

    if add_noise:
        n = np.random.normal(0, 0.05, NUM_ACTIVE_DATAPOINTS)*0.3*max_power
    else:
        n = np.zeros(NUM_ACTIVE_DATAPOINTS)

    skewnorm_ = skewnorm.pdf(x, SKEWING_FACTOR, loc = 1, scale=1.5)

    P_L = max_power * skewnorm_ + min_power + n
    t = np.linspace(0, days*HOURS, num= days*HOURS*SAMPLES_PER_HOUR)

    P_L = np.concatenate( ([P_L for _ in range(days)]), axis = None )
    if plot:
        # Plot 
        plt.figure()
        plt.plot(t, P_L)

        plt.xlabel('Time [h]')
        plt.ylabel('P_L [kW]')
        plt.title('P_load simulation')
    
    return P_L

if __name__ == "__main__":
    simulate_p_load(days = 3)
    plt.show()