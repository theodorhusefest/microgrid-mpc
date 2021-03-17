import time
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from components.loads import Load
from ocp.scenario import ScenarioOCP
from components.battery import Battery
from components.spot_price import get_spot_price
from utils.scenario_tree import build_scenario_tree, get_scenarios


def scenario_mpc():
    """
    Main function for mpc-scheme with receding horizion.
    """
    conf = utils.parse_config()
    datafile = conf["datafile"]
    loads_trainfile = conf["loads_trainfile"]
    data = pd.read_csv("./data/data_oct20.csv", parse_dates=["date"]).iloc[::10]

    logpath = None
    log = input("Log this run? ")

    if log in ["y", "yes", "Yes"]:
        foldername = input("Enter logfolder name? (enter to skip) ")
        logpath = utils.create_logs_folder(conf["logpath"], foldername)

    openloop = conf["openloop"]
    perfect_predictions = conf["perfect_predictions"]

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour
    Nr = conf["robust_horizon"]
    branch_factor = conf["branch_factor"]
    N_sim = 100
    N_scenarios = branch_factor ** Nr

    start_time = time.time()
    step_time = start_time

    pv, pv_pred, l1, l1_pred, l2, l2_pred, grid_buy = utils.load_data()

    l = Load(N, loads_trainfile, "L", groundtruth=l1 + l2)
    E = np.ones(144) * 1  # get_spot_price()
    B = Battery(T, N, **conf["battery"])

    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    pv_measured = []
    l1_measured = []
    l2_measured = []

    # Build reference tree
    tree, leaf_nodes = build_scenario_tree(
        N, Nr, branch_factor, np.ones(N + 1), 0, np.ones(N + 1), 0
    )

    ocp = ScenarioOCP(T, N, N_scenarios)
    s_data = ocp.s_data(0)

    s0, lbs, ubs, lbg, ubg = ocp.build_scenario_ocp()

    sys_metrics = metrics.SystemMetrics()

    for step in range(simulation_horizon - N):

        # Get measurements
        pv_true = pv[step]
        l_true = l.get_measurement(step)

        # Get predictions
        pv_ref = pv[step : step + N + 1]
        l_ref = l.scaled_mean_pred(l_true, step)
        root, leaf_nodes = build_scenario_tree(
            N, Nr, branch_factor, pv_ref, 0.001, l_ref, 0.1
        )

        pv_scenarios = get_scenarios(leaf_nodes, "pv")
        l_scenarios = get_scenarios(leaf_nodes, "l")

        # Update parameters
        for i in range(N_scenarios):
            s0["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            lbs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            ubs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)

            for k in range(N):
                s_data["scenario" + str(i), "data", k, "pv"] = pv_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "l"] = l_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "E"] = 1

        xk_opt, Uk_opt = ocp.solve_nlp([s0, lbs, ubs, lbg, ubg], s_data)

        # Simulate the system after disturbances
        e, uk = utils.calculate_real_u(
            xk_opt, Uk_opt, pv[step + 1], l.get_measurement(step + 1)
        )

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        sys_metrics.update_metrics(
            [Pbc[step], Pbd[step], Pgb[step], Pgs[step]], E[step]
        )

        utils.print_status(step, [B.get_SOC(openloop)], step_time, every=50)
        step_time = time.time()

    sys_metrics.calculate_consumption_rate(Pgs, pv_measured)
    sys_metrics.calculate_dependency_rate(Pgb, l.true)
    sys_metrics.print_metrics()

    # Plotting
    u = np.asarray(
        [np.asarray(Pbc) - np.asarray(Pbd), np.asarray(Pgb) - np.asarray(Pgs)]
    )

    if openloop:
        p.plot_control_actions(
            np.asarray([ocp.Pbc - ocp.Pbd, ocp.Pgb - ocp.Pgb]),
            horizon - T,
            actions_per_hour,
            logpath,
            legends=["Battery", "Grid"],
            title="Optimal Control Actions",
        )

    else:
        p.plot_control_actions(
            u,
            horizon - T,
            actions_per_hour,
            logpath,
            legends=["Battery", "Grid"],
            title="Simulated Control Actions",
        )

    p.plot_data(
        np.asarray([B.x_sim, B.x_opt]),
        title="State of charge",
        legends=["SOC", "SOC_opt"],
    )

    p.plot_data(np.asarray([pv]), title="PV", legends=["PV"])

    p.plot_data(np.asarray([l.true]), title="Loads", legends=["l"])

    p.plot_data(np.asarray([E]), title="Spot Prices", legends=["Spotprice"])

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    plt.ion()
    if True:
        plt.show(block=True)


if __name__ == "__main__":
    scenario_mpc()
