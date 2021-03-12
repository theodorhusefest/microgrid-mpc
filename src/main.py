import time
import numpy as np
import pandas as pd
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from pprint import pprint

from components.spot_price import get_spot_price
from ocp.nominel_struct import NominelMPC
from components.loads import Load
from components.battery import Battery


def main():
    """
    Main function for mpc-scheme with receding horizion.
    """
    conf = utils.parse_config()
    datafile = conf["datafile"]
    loads_trainfile = conf["loads_trainfile"]

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

    ocp = NominelMPC(T, N)
    sys_metrics = metrics.SystemMetrics()

    x, lbx, ubx, lbg, ubg = ocp.build_nlp()

    for step in range(simulation_horizon - N):

        # Get measurements
        x["states", 0, "SOC"] = B.get_SOC(openloop)
        lbx["states", 0, "SOC"] = B.get_SOC(openloop)
        ubx["states", 0, "SOC"] = B.get_SOC(openloop)

        pv_true = pv[step]
        l1_true = l1.get_measurement(step)
        l2_true = l2.get_measurement(step)

        pv_measured.append(pv_true)
        l1_measured.append(l1_true)
        l2_measured.append(l2_true)

        # Create predictions for next period
        if perfect_predictions:
            pv_ref = pv[step + 1 : step + N + 1]
            l1_ref = l1.perfect_pred(step)
            l2_ref = l2.perfect_pred(step)
            E_ref = E[step : step + N]
        else:
            pv_ref = pv_pred[step + 1 : step + N + 1]
            l1_ref = l1.scaled_mean_pred(l1_true, step)
            l2_ref = l2.scaled_mean_pred(l2_true, step)
            E_ref = E[step : step + N]

        forecasts = ocp.update_forecasts(pv_ref, l1_ref, l2_ref, E_ref)

        xk_opt, Uk_opt = ocp.solve_nlp([x, lbx, ubx, lbg, ubg], forecasts)

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
    print(ocp.Pbc.shape)
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
