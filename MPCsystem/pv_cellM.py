""" 
Simulation module for PV-cell 
Based on the real signals that looks somewhat like a sinus with max-value around 75 kW
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def simulate_pv_cell(
    resulution= 10,
    max_power= 20,
    sunrise= 8,
    sunset= 16,
    days = 1,
    add_noise = True,
    plot = True):
    """ Simulation of PV-cell
    
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
    ACTIVE_HOURS= sunset-sunrise
    SAMPLES_PER_HOUR= int(60/resulution)
    NUM_ACTIVE_DATAPOINTS= SAMPLES_PER_HOUR*ACTIVE_HOURS
    SKEWING_FACTOR = -3

    x = np.linspace(skewnorm.ppf(0.1, SKEWING_FACTOR),
                skewnorm.ppf(0.99, SKEWING_FACTOR), NUM_ACTIVE_DATAPOINTS)

    if add_noise:
        n = np.random.normal(0, 0.05, NUM_ACTIVE_DATAPOINTS)
        n[::2] = 0
    else:
        n = np.zeros(NUM_ACTIVE_DATAPOINTS)
    P_pv_final = []
    skewnorm_ = skewnorm.pdf(x, SKEWING_FACTOR, loc = 0, scale=0.5)
    t_pre_sunset = np.zeros(sunrise*SAMPLES_PER_HOUR)
    t_post_sunset = np.zeros((HOURS-sunset)*SAMPLES_PER_HOUR)
    t = np.linspace(0, days*HOURS, num=days*HOURS*SAMPLES_PER_HOUR)
    for i in range(0, days):
        Weather = np.random.randint(0, 10)
        if Weather < 3:  # Rain or fully cloudy
            pv_values = max_power * 0.2 * np.clip((skewnorm_ + n), 0, np.inf)
        elif Weather > 7:  # Partially cloudy
            pv_values = max_power * np.clip((skewnorm_ + n), 0, np.inf)
            pv_values = np.multiply(pv_values, np.random.rand(ACTIVE_HOURS * SAMPLES_PER_HOUR))
        else:  # Sunny day
            pv_values = max_power * np.clip((skewnorm_ + n), 0, np.inf)
        P_pv = np.concatenate((t_pre_sunset, pv_values, t_post_sunset), axis = None)
        P_pv_final = np.concatenate((P_pv_final, P_pv), axis=None)

    if plot:
        # Plot 
        plt.figure(12)
        plt.plot(t, P_pv_final)

        plt.xlabel('Time [h]')
        plt.ylabel('kW')
        plt.title('Simulated PV-Cell')
    
    return P_pv_final

if __name__ == "__main__":
    simulate_pv_cell(days = 3)
    plt.show()
