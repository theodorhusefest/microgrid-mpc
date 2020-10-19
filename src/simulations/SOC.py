import numpy as np
from system import get_integrator


# def simulate_SOC(x, u0, u1, u2, u3, PV_pred, PL_pred, PV, PL, C_MAX=700):
#     """
#     Finds the next state of charge based on the difference in predictions and measurements
#     Assumptions:
#     - Errors in prediction vs measured values are compansated in battery.
#     """
#     # timesteps = len(u0)
#     # P_bat = 0
#     # for k in range(timesteps):
#     #    e_PV = PV[k] - PV_pred[k]
#     #    e_PL = PL[k] - PL_pred[k]
#     #    print("Error PV {}, error PL {}, u0 {}, u1 {}".format(e_PV, e_PL, u0[k], u1[k]))
#     #    P_bat += e_PV - e_PL
#     #    print("P_bat", P_bat)
#     e_PV = np.mean(PV) - np.mean(PV_pred)
#     e_PL = np.mean(PL) - np.mean(PL_pred)
#     P_bat = e_PV - e_PL
#     print("u0:", u0)
#     print("u1:", u1)
#     print("u2:", u2)
#     print("u3:", u3)
#     print("PL", PL)
#     print("PV", PV)
#     print("PV_pred", PV_pred)
#     print("PL_pred", PL_pred)

#     f = get_integrator(1, len(u0), x)

#     print("Surplus/minus energy is {} kW.".format(np.around(P_bat, 2)))
#     print(x + (0.5 * P_bat / C_MAX))
#     return x + (0.5 * P_bat / C_MAX)
