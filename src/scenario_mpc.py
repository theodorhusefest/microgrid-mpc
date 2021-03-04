import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ocp.scenario import ScenarioMPC
from components.loads import Load

from monte_carlo import get_monte_carlo_scenarios

import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from pprint import pprint

from components.spot_price import get_spot_price
from ocp.scenario import ScenarioMPC
from components.loads import Load
from components.battery import Battery


def main():
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
    N_scenarios = 6
    N_sim = 100

    start_time = time.time()
    step_time = start_time

    pv, pv_pred, l1, l1_pred, l2, l2_pred, grid_buy = utils.load_data()

    l1 = Load(N, loads_trainfile, "L1", groundtruth=l1)
    l2 = Load(N, loads_trainfile, "L2", groundtruth=l2)
    E = np.ones(144) * 1  # get_spot_price()
    B = Battery(T, N, **conf["battery"])

    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    pv_measured = []
    l1_measured = []
    l2_measured = []

    ocp = ScenarioMPC(T, N, N_scenarios)

    sys_metrics = metrics.SystemMetrics()

    for step in range(simulation_horizon - N):

        # Get measurements
        pv_true = pv[step]
        l1_true = l1.get_measurement(step)
        l2_true = l2.get_measurement(step)

        pv_measured.append(pv_true)
        l1_measured.append(l1_true)
        l2_measured.append(l2_true)

        l1_scenarios = get_monte_carlo_scenarios(
            l1_true, step, N, N_sim, N_scenarios, l1, l1.scaled_mean_pred, data
        )

        l2_scenarios = get_monte_carlo_scenarios(
            l2_true, step, N, N_sim, N_scenarios, l2, l2.scaled_mean_pred, data
        )
        s = ocp.scenarios(0)
        for i in range(N_scenarios):
            for k in range(N):
                s["scenario" + str(i), "data", k, "pv"] = pv[step + k + 1]
                s["scenario" + str(i), "data", k, "l1"] = l1_scenarios[i][k]
                s["scenario" + str(i), "data", k, "l1"] = l2_scenarios[i][k]
                s["scenario" + str(i), "data", k, "E"] = 1

        xk_opt, Uk_opt = ocp.solve_scenario_tree(B.get_SOC(openloop), s)

        # Simulate the system after disturbances
        uk = utils.calculate_real_u(
            xk_opt,
            Uk_opt,
            pv[step + 1],
            l1.get_measurement(step + 1) + l2.get_measurement(step + 1),
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
    sys_metrics.calculate_dependency_rate(Pgb, l1.true + l2.true)
    sys_metrics.print_metrics()

    # Plotting
    u = np.asarray(
        [np.asarray(Pbc) - np.asarray(Pbd), np.asarray(Pgb) - np.asarray(Pgs)]
    )
    if openloop:
        p.plot_control_actions(
            np.asarray(
                [nominal_ocp.Pbc - nominal_ocp.Pbd, nominal_ocp.Pgb - nominal_ocp.Pgb]
            ),
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

    p.plot_data(np.asarray([pv_measured]), title="PV", legends=["PV"])

    p.plot_data(np.asarray([l1.true, l2.true]), title="Loads", legends=["l1", "l2"])

    p.plot_data(np.asarray([E]), title="Spot Prices", legends=["Spotprice"])

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    plt.ion()
    if True:
        plt.show(block=True)


if __name__ == "__main__":
    main()
