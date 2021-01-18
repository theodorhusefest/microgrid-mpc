import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, Function
from utils.helpers import parse_config


def simulate_SOC(x, u_opt, PV, PL, F):
    pv = PV.copy()
    l = PL.copy()
    uk_sim = get_real_u(x, u_opt, pv, l)
    Fk = F(x0=x, p=uk_sim)

    xk_sim = Fk["xf"].full().flatten()[0]
    return xk_sim, uk_sim


def get_real_u(x, u_opt, pv, l):
    """
    Calculates the real inputs based on topology contraint
    """

    u = np.asarray([u_[0] for u_ in u_opt])
    e = -u[0] + u[1] + u[2] - u[3] + pv - l

    if is_zero(e):
        return u

    if e > 0:  # Suplus of energy
        # Can either charge or sell.
        if np.around(u[0]) != 0 and np.around(u[1]) == 0:
            u[0] += e
        elif np.around(u[3]) != 0 and np.around(u[2]) == 0:
            u[3] += e
        elif np.around(u[1]) != 0 and u[1] > e:
            u[1] -= e  # Discharge less
        elif u[2] > e:
            u[2] -= e  # Buy less
        else:
            u[2] = 0  # Stop buying
            u[3] += e  # Start selling
    else:  # Need more energy
        # Can discharge or buy

        if np.around(u[1]) != 0 and np.around(u[0]) == 0:
            u[1] -= e
        elif np.around(u[2]) != 0 and np.around(u[3]) == 0:
            u[2] -= e
        elif np.around(u[0]) != 0 and u[0] > np.abs(e):
            u[0] += e  # Charge less
        else:
            u[3] = 0  # Stop selling
            u[2] -= e  # Buy more"""

    return u


def is_zero(x):
    return np.around(x) == 0