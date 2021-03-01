import numpy as np
from ocp.nominel_struct import NominelMPC
from casadi import vertcat

import matplotlib.pyplot as plt

N = 6
ocp = NominelMPC(1, N)

w, lbw, ubw, ubg, lbg = ocp.build_nlp()

w["states", 0, "SOC"] = 0.4
lbw["states", 0, "SOC"] = 0.4
ubw["states", 0, "SOC"] = 0.4


pv = np.ones(N) * 100
l1 = np.ones(N) * 15
l2 = np.ones(N) * 25
E = np.ones(N) * 0.5


data_struct = ocp.update_forecasts(pv, l1, l2, E)


xk_opt, Uk_opt = ocp.solve_nlp([w, lbw, ubw, lbg, ubg], data_struct)

plt.plot(range(N), xk_opt)
plt.show()
