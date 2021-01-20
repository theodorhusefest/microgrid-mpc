import time
import numpy as np
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils
from ocp.linear import LinearOCP

# from simulations.simulate_SOC import simulate_SOC


def main():
    """
    Main function for mpc-scheme with receding horizion.
    """
    conf = utils.parse_config()

    logpath = None
    log = input("Do you wish to log this run? ")

    if log in ["y", "yes", "Yes"]:
        foldername = input("Do you wish to name logfolder? (enter to skip)")
        logpath = utils.create_logs_folder(conf["logpath"], foldername)

    openloop = True

    predictions = conf["predictions"]
    print("Using {} predictions.".format(predictions))

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    start_time = time.time()
    step_time = start_time

    pv, pv_pred, l1, l1_pred, l2, l2_pred, grid_buy = utils.load_data()

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour

    xk_0 = conf["x0_initial"]
    xk_1 = conf["x1_initial"]
    # xk_sim = conf["x_initial"]
    x0_opt = np.asarray([xk_0])
    x1_opt = np.asarray([xk_1])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    # u3 = np.asarray([])

    T0 = []
    T1 = []
    T2 = []

    solver = LinearOCP(T, N)

    x, lbx, ubx, lbg, ubg = solver.build_nlp()

    net_cost_grid = 0
    net_cost_bat = 0
    J = 0

    # wt_pred = [30]
    # pv_pred = [pv[0]]
    # l1_pred = [l1[0]]
    # l2_pred = [l2[0]]

    pv_error = []
    l_error = []

    for step in range(simulation_horizon - N):
        # Update NLP parameters
        x[0] = xk_0
        lbx[0] = xk_0
        ubx[0] = xk_0
        x[1] = xk_1
        lbx[1] = xk_1
        ubx[1] = xk_1

        pv_true = pv[step : step + N]
        l1_true = l1[step : step + N]
        l2_true = l2[step : step + N]

        wt_ref = 30 * np.ones(N)
        pv_ref = np.zeros(N)  # pv_true
        l1_ref = l1_true
        l2_ref = l2_true

        # pv_pred.append(pv_ref[1])
        # l_pred.append(l_ref[1])

        xk_opt, Uk_opt, Tk_opt, J_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(wt_ref, pv_ref, l1_ref, l2_ref)
        )
        J += J_opt

        T0.append(Tk_opt[0][0])
        T1.append(Tk_opt[1][0])
        T2.append(Tk_opt[2][0])
        """
        xk_sim, Uk_sim = simulate_SOC(
            xk_sim,
            Uk_opt,
            pv[step],
            l[step],
            solver.F,
        )

        x_sim = np.append(x_sim, xk_sim)
        """

        if openloop:
            xk_0 = xk_opt[0][0]  # xk is optimal
            xk_1 = xk_opt[1][0]
        # else:
        #    xk = xk_sim

        x0_opt = np.append(x0_opt, xk_0)
        x1_opt = np.append(x1_opt, xk_1)

        uk = [u[0] for u in Uk_opt]
        u0 = np.append(u0, uk[0])
        u1 = np.append(u1, uk[1])
        u2 = np.append(u2, uk[2])

        if step % 50 == 0:
            print(
                "\nFinshed iteration step {}. Current step took {}s".format(
                    step, np.around(time.time() - step_time, 2)
                )
            )
            print(
                "xsim {}%, x_opt {},{}%".format(
                    np.around(0.5, 2),
                    np.around(xk_opt[0][0], 2),
                    np.around(xk_opt[1][0], 2),
                )
            )
            step_time = time.time()

    # Plotting
    u = np.asarray([u0, u1, u2])

    p.plot_control_actions(
        u, horizon - T, actions_per_hour, logpath, legends=["Pb1", "Pb2", "Pg"]
    )

    p.plot_SOC(x0_opt, horizon - T, logpath)
    p.plot_SOC(x1_opt, horizon - T, logpath)

    p.plot_data(
        np.asarray([T0, T1, T2]), title="Topology Variables", legends=["T", "B", "L"]
    )

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    plt.show(block=True)
    plt.ion()
    plt.close("all")


if __name__ == "__main__":
    main()
