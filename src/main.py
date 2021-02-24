import time
import numpy as np
import pandas as pd
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from components.spot_price import get_spot_price
from ocp.nominel import NominelMPC
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
    E = get_spot_price()
    B = Battery(T, N, **conf["battery"])

    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    pv_measured = []
    l1_measured = []
    l2_measured = []

    solver = NominelMPC(T, N)
    sys_metrics = metrics.SystemMetrics()

    x, lbx, ubx, lbg, ubg = solver.build_nlp()

    for step in range(simulation_horizon - N):
        # Update NLP parameters
        x[0] = B.get_SOC()
        lbx[0] = B.get_SOC()
        ubx[0] = B.get_SOC()

        pv_true = pv[step : step + N]
        l1_true = l1.get_groundtruth(step)
        l2_true = l2.get_groundtruth(step)

        pv_ref = pv_true
        l1_ref = l1.scaled_mean_pred(l1_true[1], step)
        l2_ref = l2.scaled_mean_pred(l2_true[1], step)
        E_ref = E[step : step + N]

        pv_measured.append(pv_true[0])
        l1_measured.append(l1_true[0])
        l2_measured.append(l2_true[0])

        xk_opt, Uk_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(pv_ref, l1_ref, l2_ref, E_ref)
        )
        uk = [u[0] for u in Uk_opt]
        B.set_x(xk_opt[1])
        # Uk_sim, Tk_sim = utils.calculate_real_u(
        #    uk, Tk, wt_true[0], pv_true[0], l1_true[0], l2_true[0]
        # )

        # B.simulate_SOC(xk_opt[0][0], [uk[0], uk[1]])

        u0 = np.append(u0, uk[0])
        u1 = np.append(u1, uk[1])
        u2 = np.append(u2, uk[2])
        u3 = np.append(u3, uk[3])

        # sys_metrics.update_metrics([u0[step], u1[step], u2[step], u3[step]], E[step])

        utils.print_status(step, [B.get_SOC()], step_time, every=50)
        step_time = time.time()

    # sys_metrics.print_metrics()

    # Plotting
    u = np.asarray([u0 - u1, u2 - u3])
    p.plot_control_actions(
        u, horizon - T, actions_per_hour, logpath, legends=["Battery", "Grid"]
    )

    p.plot_data(
        np.asarray([B.x_sim]),
        title="State of charge",
        legends=["SOC0"],
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
