import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, Function


def simulate_SOC(x, u_opt, PV, PL, PV_pred, PL_pred, F, C_MAX=700, nb_c=0.8, nb_d=0.8):
    uk_sim = get_real_u(u_opt, PV, PL, PV_pred, PL_pred)
    Fk = F(x0=x, p=uk_sim)

    xk_sim = Fk["xf"].full().flatten()[0]
    return xk_sim, uk_sim


def get_real_u(u_opt, PV, PL, PV_pred, PL_pred):
    """
    Calculates the real inputs when there are errors between
    prediction and real PV and load values
    """
    u = np.asarray([u_[0] for u_ in u_opt])
    e_PV = PV[0] - PV_pred[0]
    e_PL = PL[0] - PL_pred[0]
    e_Pbat = e_PV - e_PL
    if e_Pbat > 0:
        u[0] += e_Pbat
    else:
        u[1] -= e_Pbat
    return u