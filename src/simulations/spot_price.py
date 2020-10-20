import numpy as np

HOURS_PER_DAY = 24


def simulate_spotprice(days, samples_per_hour, start_price=15):
    """
    Returns a simulation of the spotprices
    """

    return np.full(days * HOURS_PER_DAY * samples_per_hour, 15)
