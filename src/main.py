import time
import numpy as np
import pandas as pd
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from components.spot_price_test import get_spot_price_test
from utils.viz import GraphViz
from ocp.linear import LinearOCP
from components.windturbine import WindTurbine
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
    B1 = Battery(T, N, **conf["components"]["B1"])
    B2 = Battery(T, N, **conf["components"]["B1"])
    B3 = Battery(T, N, **conf["components"]["B1"])

    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

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
        x[0] = B1.get_SOC()
        lbx[0] = B1.get_SOC()
        ubx[0] = B1.get_SOC()
        x[1] = B2.get_SOC()
        lbx[1] = B2.get_SOC()
        ubx[1] = B2.get_SOC()
        x[2] = B3.get_SOC()
        lbx[2] = B3.get_SOC()
        ubx[2] = B3.get_SOC()

        wt_true = wt.get_power(2 * np.ones(N))
        pv_true = pv[step : step + N]
        l1_true = l1.get_groundtruth(step)
        l2_true = l2.get_groundtruth(step)

        wt_ref = wt.get_power(2 * np.ones(N))  # + np.random.normal(1, 0.1, N))
        pv_ref = pv_true
        l1_ref = l1.scaled_mean_pred(l1_true[1], step)
        l2_ref = l2.scaled_mean_pred(l2_true[1], step)
        E_ref = E[step : step + N]

        wt_measured.append(wt_ref[0])
        pv_measured.append(pv_true[0])
        l1_measured.append(l1_true[0])
        l2_measured.append(l2_true[0])

        xk_opt, Uk_opt, Tk_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(wt_ref, pv_ref, l1_ref, l2_ref, E_ref)
        )
        uk = [u[0] for u in Uk_opt]
        Tk = [T[0] for T in Tk_opt]

        T0.append(Tk[0])
        T1.append(Tk[1])
        T2.append(Tk[2])

        Uk_sim, Tk_sim = utils.calculate_real_u(
            uk, Tk, wt_true[0], pv_true[0], l1_true[0], l2_true[0]
        )

        B1.simulate_SOC(xk_opt[0][0], [uk[0], uk[1]])
        B2.simulate_SOC(xk_opt[1][0], [uk[2], uk[3]])
        B3.simulate_SOC(xk_opt[2][0], [uk[4], uk[5]])

        u0 = np.append(u0, uk[0] - uk[1])
        u1 = np.append(u1, uk[2] - uk[3])
        u2 = np.append(u2, uk[4] - uk[5])
        u3 = np.append(u3, uk[6] - uk[7])

        sys_metrics.update_metrics([u0[step], u1[step], u2[step], u3[step]], E[step])

        utils.print_status(
            step, [B1.get_SOC(), B2.get_SOC(), B3.get_SOC()], step_time, every=50
        )
        step_time = time.time()

    sys_metrics.print_metrics()

    # Plotting
    u = np.asarray([u0, u1, u2, u3])
    p.plot_control_actions(
        u, horizon - T, actions_per_hour, logpath, legends=["Pb1", "Pb2", "Pb3", "Pg"]
    )

    p.plot_data(
        np.asarray([B1.x_sim, B2.x_sim, B3.x_sim]),
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
            B1.x_sim,
            B2.x_sim,
            B3.x_sim,
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
    if True:

        plt.show(block=True)


if __name__ == "__main__":
    main()
