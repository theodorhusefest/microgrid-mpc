import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, Function
from utils.helpers import parse_config


def simulate_SOC(x, u_opt, PV, PL, PV_pred, PL_pred, F):
    uk_sim = get_real_u(x, u_opt, PV, PL, PV_pred, PL_pred)
    Fk = F(x0=x, p=uk_sim)

    xk_sim = Fk["xf"].full().flatten()[0]
    return xk_sim, uk_sim


def get_real_u(x, u_opt, PV, PL, PV_pred, PL_pred):
    """
    Calculates the real inputs when there are errors between
    prediction and real PV and load values
    """
    u = np.asarray([u_[0] for u_ in u_opt])
    e_PV = PV[0] - PV_pred[0]
    e_PL = PL[0] - PL_pred[0]
    e_Pbat = e_PV - e_PL

    if e_Pbat > 0:  # Suplus of energy
        # Can either charge or sell.
        if np.around(u[0]) != 0 and np.around(u[1]) == 0:
            u[0] += e_Pbat
        elif np.around(u[3]) != 0 and np.around(u[2]) == 0:
            u[3] += e_Pbat
        elif np.around(u[1]) != 0 and u[1] > e_Pbat:
            u[1] -= e_Pbat  # Discharge less
        elif u[2] > e_Pbat:
            u[2] -= e_Pbat  # Buy less
        else:
            u[2] = 0  # Stop buying
            u[3] += e_Pbat  # Start selling
    else:  # Need more energy
        # Can discharge or buy

        if np.around(u[1]) != 0 and np.around(u[0]) == 0:
            u[1] -= e_Pbat
        elif np.around(u[2]) != 0 and np.around(u[3]) == 0:
            u[2] -= e_Pbat
        elif np.around(u[0]) != 0 and u[0] > np.abs(e_Pbat):
            u[0] += e_Pbat  # Charge less
        else:
            u[3] = 0  # Stop selling
            u[2] -= e_Pbat  # Buy more
            print(u)
            print("Somethings is wrong, e = ", e_Pbat)

    return u
