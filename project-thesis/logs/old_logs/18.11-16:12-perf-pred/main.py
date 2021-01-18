import time
from casadi import vertcat
import numpy as np
import matplotlib.pyplot as plt
import utils.plots as p
from utils.helpers import create_logs_folder, parse_config, load_datafile, save_datafile
from simulations.simulate import get_simulations
from simulations.simulate_SOC import simulate_SOC
from solver import OptiSolver


def main():
    """
    Main function for mpc-scheme with receding horizion.

    Assumptions:
    - Only get weather predicitons at start of interval

    """
    conf = parse_config()

    logpath = None
    log = input("Do you wish to log this run? ")

    if log in ["y", "yes", "Yes"]:
        foldername = input("Do you wish to name logfolder? (enter to skip)")
        logpath = create_logs_folder(conf["logpath"], foldername)

    openloop = input("Run openloop? ")

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    start_time = time.time()
    step_time = start_time

    # Get predictions for given time period
    PV, PV_pred, PL, PL_pred, grid_buy, grid_sell = get_simulations(logpath)

    print(
        "Predicted energy produced {}, predicted energy consumed {}".format(
            np.sum(PV_pred), np.sum(PL_pred)
        )
    )

    print(
        "Actual energy produced {}, actual energy consumed {}".format(
            np.sum(PV), np.sum(PL)
        )
    )
    print("Predicted energy surplus/deficit:", np.sum(PV_pred) - np.sum(PL_pred))
    print("Actual energy surplus/deficit:", np.sum(PV) - np.sum(PL))

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

    for step in range(simulation_horizon - N):
        # Update NLP parameters
        x[0] = xk
        lbx[0] = xk
        ubx[0] = xk
        pv_ref = PV_pred[step : step + N]
        pl_ref = PL_pred[step : step + N]

        xk_opt, Uk_opt = solver.solve_nlp(
            [x, lbx, ubx, lbg, ubg], vertcat(pv_ref, pl_ref)
        )
        x_opt = np.append(x_opt, xk_opt[1])

        xk_sim, Uk_sim = simulate_SOC(
            xk_sim,
            Uk_opt,
            PV[step : step + N :],
            PL[step : step + N :],
            PV_pred[step : step + N :],
            PL_pred[step : step + N :],
            solver.F,
        )
        x_sim = np.append(x_sim, xk_sim)

        if openloop in ["y", "yes", "Yes"]:
            xk = xk_opt[1]  # xk is optimal
        else:
            xk = xk_sim  # xk is simulated difference between measurements and predictions

        # Get the next control actions
        uk = [u[0] for u in Uk_opt]

        u0 = np.append(u0, uk[0])
        u1 = np.append(u1, uk[1])
        u2 = np.append(u2, uk[2])
        u3 = np.append(u3, uk[3])

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
        title="Battery Action",
        legends=["Battery Charge", "Battery Discharge"],
    )

    p.plot_control_actions(
        u_grid,
        horizon - T,
        actions_per_hour,
        logpath,
        title="Grid Action",
        legends=["Grid Buy", "Grid Sell"],
    )

    p.plot_SOC([x_opt], horizon - T, logpath)

    p.plot_data(
        [PV, PL],
        logpath=logpath,
        legends=["PV", "PL"],
        title="PV vs. PL",
        ylabel="Power [kW]",
    )

    # p.plot_data(
    #    [x_opt, x_sim],
    #    logpath=logpath,
    #    legends=["SOC optimal", "SOC simulated"],
    #    title="Simulated vs optimal SOC",
    # )

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))
    save_datafile(
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
    plt.show()


if __name__ == "__main__":
    main()
