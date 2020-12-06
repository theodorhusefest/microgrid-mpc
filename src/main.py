import time
import numpy as np
from casadi import vertcat
import matplotlib.pyplot as plt
import utils.plots as p

from arima import Arima
from solver import OptiSolver
import metrics as metrics
import utils.helpers as utils
from simulations.simulate_SOC import simulate_SOC


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

    openloop = False

    use_arima = not conf["perfect_predictions"]
    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    start_time = time.time()
    step_time = start_time

    PV, PV_pred, PL, PL_pred, grid_buy, grid_sell = utils.load_data()

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour

    xk = conf["x_inital"]
    xk_sim = conf["x_inital"]
    x_opt = np.asarray([xk])
    x_sim = np.asarray([xk])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    solver = OptiSolver(N)

    nlp_params = solver.build_nlp(
        T,
        N,
    )

    x = nlp_params[0]
    lbx = nlp_params[1]
    ubx = nlp_params[2]
    lbg = nlp_params[3]
    ubg = nlp_params[4]

    net_cost_grid = 0
    net_cost_bat = 0
    J = 0

    pv_preds = [PV[0]]
    pl_preds = [PL[0]]

    plt.figure()

    if use_arima:
        pv_model = Arima("PV")
        pl_model = Arima("PL")

    for step in range(simulation_horizon - N):
        # Update NLP parameters
        x[0] = xk
        lbx[0] = xk
        ubx[0] = xk

        if True:  # Predicted values equal to current
            pv_ref = np.ones(N) * PV[step]
            pl_ref = np.ones(N) * PL[step]

        elif use_arima:  # Estimate using ARIMA
            pv_model.update(PV[step])
            pl_model.update(PL[step])

            pv_ref = pv_model.predict(T)
            pl_ref = pl_model.predict(T)
        else:  # Use true predictions
            pv_ref = PV[step + 1 : step + N + 1]
            pl_ref = PL[step + 1 : step + N + 1]

        pv_preds.append(pv_ref[0])
        pl_preds.append(pl_ref[0])

        plt.plot(range(step, step + N), pv_ref, c="b")
        plt.plot(range(step, step + N), PV[step : step + N], c="r")

        xk_opt, Uk_opt, J_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(pv_ref, pl_ref)
        )
        J += J_opt
        x_opt = np.append(x_opt, xk_opt[1])

        xk_sim, Uk_sim = simulate_SOC(
            xk_sim,
            Uk_opt,
            PV[step + 1],
            PL[step + 1],
            solver.F,
        )
        x_sim = np.append(x_sim, xk_sim)

        if openloop:
            xk = xk_opt[1]  # xk is optimal
            uk = [u[0] for u in Uk_opt]
            u0 = np.append(u0, uk[0])
            u1 = np.append(u1, uk[1])
            u2 = np.append(u2, uk[2])
            u3 = np.append(u3, uk[3])
        else:
            xk = xk_sim  # xk is simulated difference between measurements and predictions
            u0 = np.append(u0, Uk_sim[0])
            u1 = np.append(u1, Uk_sim[1])
            u2 = np.append(u2, Uk_sim[2])
            u3 = np.append(u3, Uk_sim[3])

        net_cost_grid += metrics.net_spending_grid(Uk_sim, 1.5, actions_per_hour)
        net_cost_bat += metrics.net_cost_battery(
            Uk_sim, conf["system"]["battery_cost"], actions_per_hour
        )

        if step % 10 == 0:
            print(
                "\nFinshed iteration step {}. Current step took {}s".format(
                    step, np.around(time.time() - step_time, 2)
                )
            )
            print(
                "xsim {}%, x_opt {}%".format(
                    np.around(xk_sim, 2), np.around(xk_opt[1], 2)
                )
            )
            step_time = time.time()

        utils.check_constrain_satisfaction(
            u0[-1], u1[-1], u2[-1], u3[-1], PV[step + 1], PL[step + 1]
        )

    plt.show()
    mean_change_bat = metrics.net_change_battery(u0, u1)
    print()
    print("Net spending grid: {} kr".format(np.around(net_cost_grid, 2)))
    print("Peak power consumption: {} kW".format(np.around(np.max(u2), 2)))
    print("Net spending battery: {} kr".format(np.around(net_cost_bat, 2)))
    print("Average switching battery: {}".format(np.around(mean_change_bat, 2)))
    print(
        "Grid + battery spending:",
        np.around(net_cost_grid + net_cost_bat, 2),
    )

    E_start = conf["x_inital"] * conf["system"]["C_MAX"]
    E_end = xk * conf["system"]["C_MAX"]

    print(
        "Change in battery energy {} kr".format(np.around(1.5 * (E_end - E_start), 2))
    )
    print("Total spending:", net_cost_grid + net_cost_bat + 1.5 * (E_end - E_start))

    # Plotting
    u = np.asarray([-u0, u1, u2, -u3])
    u_bat = np.asarray([-u0, u1])
    u_grid = np.asarray([u2, -u3])

    p.plot_control_actions(u, horizon - T, actions_per_hour, logpath)

    p.plot_control_actions(
        u_bat,
        horizon - T,
        actions_per_hour,
        logpath,
        title="Battery actions",
        legends=["Battery Charge", "Battery Discharge"],
    )

    p.plot_control_actions(
        u_grid,
        horizon - T,
        actions_per_hour,
        logpath,
        title="Grid actions",
        legends=["Grid Buy", "Grid Sell"],
    )

    p.plot_SOC(x_opt, horizon - T, logpath)
    # p.plot_SOC(x_sim, horizon, logpath, title="Simulated State of Charge")

    p.plot_data(
        [x_opt, x_sim],
        logpath=logpath,
        legends=["SOC optimal", "SOC simulated"],
        title="Simulated vs optimal SOC",
    )

    p.plot_data(
        [PV, PL],
        logpath=logpath,
        legends=["PV-production", "Load Demands"],
        title="PV Production & Load Demands - Reference",
    )

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))
    utils.save_datafile(
        [x_opt, x_sim, u0, u1, u2, u3, PV, PV_pred, PL, PL_pred],
        names=[
            "x_opt",
            "x_sim",
            "u0",
            "u1",
            "u2",
            "u3",
            "PV",
            "PV_pred",
            "PL",
            "PL_pred",
        ],
        logpath=logpath,
    )
    if conf["plot_predictions"]:
        p.plot_data(
            [PV[: len(pv_preds)], pv_preds], legends=["Real PV", "Predicted PV"]
        )
        p.plot_data(
            [PL[: len(pl_preds)], pl_preds], legends=["Real Pl", "Predicted PL"]
        )
        metrics.rmse_predictions(PV, pv_preds)
        metrics.rmse_predictions(PL, pl_preds)

    plt.show(block=True)
    plt.ion()
    plt.close("all")


if __name__ == "__main__":
    main()
