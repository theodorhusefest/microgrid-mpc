import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ocp.scenario_reduced import ReducedScenarioMPC
from ocp.scenario_full import FullScenarioMPC
from components.loads import Load

from monte_carlo import get_monte_carlo_scenarios
from scenario_tree import build_scenario_tree


from pprint import pprint


start = time.time()
T = 1
N = 4
Nr = 3
Nu = 4
p = 20
branch_factor = 2


pv_base = np.ones(N + 1) * 10
l1_base = np.ones(N + 1) * 50
l2_base = np.ones(N) * 10
E_base = np.ones(N) * 1

x_initial = 0.4
step = 40

ocp = FullScenarioMPC(T, N, 8)

root = ocp.build_scenario_ocp(Nr, branch_factor)
# root.print_children()


"""
leaf_nodes = build_scenario_tree(N, Nr, branch_factor, pv_base, 0.1, l1_base, 0.1)

N_scenarios = len(leaf_nodes)
ocp = ScenarioMPC(T, N, N_scenarios)
s = ocp.s_data(0)

step_time = time.time()
print("Used {}s on building tree and ocp.".format(step_time - start))

s0, lbs, ubs, lbg, ubg = ocp.build_scenario_ocp()

pv_scenarios = get_all_scenarios(leaf_nodes, "pv")
l_scenarios = get_all_scenarios(leaf_nodes, "l")

for i in range(N_scenarios):
    s0["scenario" + str(i), "states", 0, "SOC"] = x_initial
    lbs["scenario" + str(i), "states", 0, "SOC"] = x_initial
    ubs["scenario" + str(i), "states", 0, "SOC"] = x_initial

    plt.plot(range(N + 1), pv_scenarios[i])

    for k in range(N):
        s["scenario" + str(i), "data", k, "pv"] = pv_scenarios[i][k]
        s["scenario" + str(i), "data", k, "l"] = l_scenarios[i][k]
        s["scenario" + str(i), "data", k, "E"] = E_base[k]

plt.show()
s_opt = ocp.solver(x0=s0, lbx=lbs, ubx=ubs, lbg=lbg, ubg=ubg, p=s)

s_opt = s_opt["x"].full().flatten()

s_opt = ocp.s(s_opt)
print("\n Used {}s.".format(time.time() - step_time))


def plot_input(N, N_scenarios, s, input_):
    plt.figure()
    for i in range(N_scenarios):
        plt.plot(
            range(N - 1),
            s_opt["scenario" + str(i), "inputs", :, input_],
            label="Scenario " + str(i),
        )
    plt.title(input_)
    plt.legend()


plot_input(N, N_scenarios, s_opt, "Pbc")
plot_input(N, N_scenarios, s_opt, "Pbd")
plot_input(N, N_scenarios, s_opt, "Pgb")
plot_input(N, N_scenarios, s_opt, "Pgs")


plt.figure()
for i in range(N_scenarios):
    plt.plot(range(N), s_opt["scenario" + str(i), "states", :, "SOC"])

plt.ylim([0, 1])
plt.show()


def plot_serie(N, N_scenarios, serie, title):
    plt.figure()
    for k in range(N_scenarios):
        plt.plot(range(N), serie[k], label="Scenario" + str(k))
    plt.legend()
    plt.title(title)


plot_serie(N, N_scenarios, SOC, "SOC")
plot_serie(N - 1, N_scenarios, Pbc, "Pbc")
plot_serie(N - 1, N_scenarios, Pbd, "Pbd")
plot_serie(N - 1, N_scenarios, Pgb, "Pgb")
plot_serie(N - 1, N_scenarios, Pgs, "Pgs")

plt.show()
"""