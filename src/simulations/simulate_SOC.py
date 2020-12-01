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
    conf = parse_config()["system"]
    u = np.asarray([u_[0] for u_ in u_opt])
    e_PV = PV[0] - PV_pred[0]
    e_PL = PL[0] - PL_pred[0]
    e_Pbat = e_PV - e_PL

    if x <= conf["x_min"] and e_Pbat > 0:  # Need more energy and below x_min
        print("Buying from grid")
        u[2] += e_Pbat
    elif x >= conf["x_max"] and e_Pbat < 0:  # Surplus of energy and above x_max
        print("Selling to grid")
        u[3] -= e_Pbat
    elif e_Pbat > 0:
        u[0] += e_Pbat
    else:
        u[1] -= e_Pbat
    return u