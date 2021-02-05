import time
import numpy as np
import pandas as pd
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from profiles.spot_price_test import get_spot_price_test
from utils.viz import GraphViz
from ocp.linear import LinearOCP
from profiles.windturbine import WindTurbine
from profiles.loads import Load

# from simulations.simulate_SOC import simulate_SOC


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

    openloop = True

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour

    start_time = time.time()
    step_time = start_time

    pv, pv_pred, l1, l1_pred, l2, l2_pred, grid_buy = utils.load_data()

    l1 = Load(N, loads_trainfile, "L1", groundtruth=datafile)
    l2 = Load(N, loads_trainfile, "L2", groundtruth=datafile)
    wt = WindTurbine()
    E = get_spot_price_test()

    xk_0 = conf["x0_initial"]
    xk_1 = conf["x1_initial"]
    xk_2 = conf["x2_initial"]
    x0_opt = np.asarray([xk_0])
    x1_opt = np.asarray([xk_1])
    x2_opt = np.asarray([xk_2])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])
    u4 = np.asarray([])
    u5 = np.asarray([])
    u6 = np.asarray([])
    u7 = np.asarray([])

    wt_measured = []
    pv_measured = []
    l1_measured = []
    l2_measured = []

    T0 = []
    T1 = []
    T2 = []

    solver = LinearOCP(T, N)
    sys_metrics = metrics.SystemMetrics()

    x, lbx, ubx, lbg, ubg = solver.build_nlp()

    for step in range(simulation_horizon - N):
        # Update NLP parameters
        x[0] = xk_0
        lbx[0] = xk_0
        ubx[0] = xk_0
        x[1] = xk_1
        lbx[1] = xk_1
        ubx[1] = xk_1
        x[2] = xk_2
        lbx[2] = xk_2
        ubx[2] = xk_2

        pv_true = pv[step : step + N]
        l1_true = l1.get_groundtruth(step)
        l2_true = l2.get_groundtruth(step)

        wt_ref = wt.get_power(3 * np.ones(N) + np.random.normal(1, 0.1, N))
        pv_ref = pv_true
        l1_ref = l1_true  # l1.get_scaled_mean(l1_true[0], step)
        l2_ref = l2_true  # l2.get_scaled_mean(l2_true[0], step)
        E_ref = E[step : step + N]

        wt_measured.append(wt_ref[0])
        pv_measured.append(pv_ref[0])
        l1_measured.append(l1_ref[0])
        l2_measured.append(l2_ref[0])

        xk_opt, Uk_opt, Tk_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(wt_ref, pv_ref, l1_ref, l2_ref, E_ref)
        )

        T0.append(Tk_opt[0][0])
        T1.append(Tk_opt[1][0])
        T2.append(Tk_opt[2][0])

        if openloop:
            xk_0 = xk_opt[0][0]  # xk is optimal
            xk_1 = xk_opt[1][0]
            xk_2 = xk_opt[2][0]
        # else:
        #    xk = xk_sim

        x0_opt = np.append(x0_opt, xk_0)
        x1_opt = np.append(x1_opt, xk_1)
        x2_opt = np.append(x2_opt, xk_2)

        u0 = np.append(u0, Uk_opt[0][0] - Uk_opt[1][0])
        u1 = np.append(u1, Uk_opt[2][0] - Uk_opt[3][0])
        u2 = np.append(u2, Uk_opt[4][0] - Uk_opt[5][0])
        u3 = np.append(u3, Uk_opt[6][0] - Uk_opt[7][0])

        sys_metrics.update_metrics([u0[step], u1[step], u2[step], u3[step]], E[step])

        utils.print_status(step, [xk_0, xk_1, xk_2], step_time, every=50)
        step_time = time.time()

    sys_metrics.print_metrics()

    # Plotting
    u = np.asarray([u0, u1, u2, u3])
    p.plot_control_actions(
        u, horizon - T, actions_per_hour, logpath, legends=["Pb1", "Pb2", "Pb3", "Pg"]
    )

    p.plot_data(
        np.asarray([x0_opt, x1_opt, x2_opt]),
        title="State of charge",
        legends=["SOC0", "SOC1", "SOC2"],
    )

    p.plot_data(
        np.asarray([T0, T1, T2]), title="Topology Variables", legends=["T", "B", "L"]
    )

    p.plot_data(
        np.asarray([wt_measured, pv_measured]), title="Renewables", legends=["wt", "pv"]
    )

    p.plot_data(
        np.asarray([l1_measured, l2_measured]), title="Loads", legends=["l1", "l2"]
    )

    p.plot_data(np.asarray([E]), title="Spot Prices")

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    data = utils.create_datafile(
        [
            x0_opt,
            x1_opt,
            x2_opt,
            u0,
            u1,
            u2,
            u3,
            wt_measured,
            pv_measured,
            l1_measured,
            l2_measured,
            T0,
            T1,
            T2,
        ],
        names=[
            "SOC1",
            "SOC2",
            "SOC3",
            "PB1",
            "PB2",
            "PB3",
            "PG",
            "WT",
            "PV",
            "L1",
            "L2",
            "T",
            "B",
            "L",
        ],
    )

    plt.ion()

    g = GraphViz(figsize=(20, 10))
    g.plot_with_slider(data.drop(["SOC1", "SOC2", "SOC3"], axis=1))

    plt.show(block=True)


if __name__ == "__main__":
    main()
