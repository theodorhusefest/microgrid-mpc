import time
from casadi import *
import matplotlib.pyplot as plt
import utils.plots as p
from utils.helpers import create_logs_folder, parse_config, load_datafile
from simulations.simulate import get_simulations
from open_loop import open_loop_optimization


def main():
    """
    Main function for mpc-scheme with receding horizion.

    Assumptions:
    - Only get weather predicitons at start of interval

    """
    conf = parse_config()

    logpath = None
    log = input("Do you wish to log this run? ")

    if log == "y" or log == "yes" or log == "Yes":
        logpath = create_logs_folder(conf["logpath"])

    openloop = input("Run only openloop? ")

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["days"] * 24  # timehorizon

    start = time.time()
    step_time = start

    # Get predictions for time period
    PV, PV_pred, PL, PL_pred, grid_buy, grid_sell = get_simulations(
        actions_per_hour,
        conf["days"],
        conf["simulations"],
        conf["datafile"],
        logpath,
        perfect_predictions=conf["perfect_predictions"],
        plot=conf["plot_predictions"],
    )

    xk = conf["x_inital"]
    x = np.asarray([xk])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    for hour in range(horizon):
        T = horizon - hour
        N = T * actions_per_hour
        xk, u, x_opt, U_opt = open_loop_optimization(
            xk,
            T,
            N,
            PV[hour * actions_per_hour : :],
            PL[hour * actions_per_hour : :],
            PV_pred[hour * actions_per_hour : :],
            PL_pred[hour * actions_per_hour : :],
            **conf["system"],
            plot=False,
            grid_buy=grid_buy[hour * actions_per_hour : :],
            grid_sell=grid_sell[hour * actions_per_hour : :],
        )

        if openloop in ["y", "yes", "Yes"]:
            x = x_opt
            u0 = U_opt[0]
            u1 = U_opt[1]
            u2 = U_opt[2]
            u3 = U_opt[3]
            break

        if conf["perfect_predictions"]:
            xk = x_opt[actions_per_hour]

        # Get the next control actions
        uk = [u[0:actions_per_hour] for u in U_opt]

        u0 = np.append(u0, uk[0])
        u1 = np.append(u1, uk[1])
        u2 = np.append(u2, uk[2])
        u3 = np.append(u3, uk[3])

        x = np.append(x, xk)

        print(
            "\nFinshed iteration hour {}. Current step took {}s".format(
                hour, np.around(time.time() - step_time, 2)
            )
        )
        step_time = time.time()

    # Plotting
    u = np.asarray([-u0, u1, u2, -u3])
    u_bat = np.asarray([-u0, u1])
    u_grid = np.asarray([u2, -u3])

    p.plot_control_actions(u, horizon, actions_per_hour, logpath)

    p.plot_control_actions(
        u_bat,
        horizon,
        actions_per_hour,
        logpath,
        title="Battery actions",
        legends=["Battery Charge", "Battery Discharge"],
    )

    p.plot_control_actions(
        u_grid,
        horizon,
        actions_per_hour,
        logpath,
        title="Grid actions",
        legends=["Grid Buy", "Grid Sell"],
    )

    p.plot_SOC(x, horizon, logpath)

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start, 2)))
    plt.show()


if __name__ == "__main__":
    main()