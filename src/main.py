import time
from casadi import *
import matplotlib.pyplot as plt
import utils.plots as p
from utils.helpers import create_logs_folder, parse_config, load_datafile

from open_loop import open_loop_optimization
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load
from simulations.spot_price import simulate_spotprice


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

    days = conf["days"]
    action_per_hour = conf["action_per_hour"]
    horizon = days * 24  # timehorizon

    start = time.time()
    step_time = start

    # Get prediction for time period
    pv_conf = conf["pv-cell"]
    PV_pred = simulate_pv_cell(
        samples_per_hour=action_per_hour,
        sunrise=2,
        sunset=20,
        max_power=pv_conf["max_power"],
        days=days,
        plot=False,
        logpath=logpath,
        add_noise=pv_conf["add_noise"],
    )

    pl_conf = conf["p-load"]
    PL_pred = simulate_p_load(
        samples_per_hour=action_per_hour,
        max_power=pl_conf["max_power"],
        days=days,
        plot=False,
        logpath=logpath,
        add_noise=pl_conf["add_noise"],
    )

    if conf["datafile"]:
        PV, PL, spotprice = load_datafile(conf["datafile"])
        grid_buy = spotprice
        grid_sell = spotprice
        p.plot_data(
            [PV, PV_pred],
            logpath=logpath,
            title="Predicted vs real PV",
            legends=["PV", "Predicted PV"],
        )
        p.plot_data(
            [PL, PL_pred],
            logpath=logpath,
            title="Predicted vs real load",
            legends=["PL", "Predicted PL"],
        )

    else:
        grid_buy = simulate_spotprice(
            days, action_per_hour, start_price=conf["grid_buy"]
        )
        grid_buy = simulate_spotprice(
            days, action_per_hour, start_price=conf["grid_sell"]
        )

    xk = conf["x_inital"]
    x = np.asarray([xk])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    for hour in range(horizon):
        print(
            "Spotprice for hour {} is {}".format(
                hour, spotprice[hour * action_per_hour]
            )
        )
        T = horizon - hour
        N = T * action_per_hour
        x_sim, u, x_opt, U_opt = open_loop_optimization(
            xk,
            T,
            N,
            PV[hour * action_per_hour : :],
            PL[hour * action_per_hour : :],
            PV_pred[hour * action_per_hour : :],
            PL_pred[hour * action_per_hour : :],
            **conf["system"],
            plot=False,
            grid_buy=grid_buy[hour * action_per_hour : :],
            grid_sell=grid_sell[hour * action_per_hour : :],
        )

        if openloop in ["y", "yes", "Yes"]:
            x = x_opt
            u0 = U_opt[0]
            u1 = U_opt[1]
            u2 = U_opt[2]
            u3 = U_opt[3]
            break

        xk = x_sim

        # Get the next control actions
        uk = [u[0:action_per_hour] for u in U_opt]

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

    p.plot_control_actions(u, horizon, action_per_hour, logpath)

    p.plot_control_actions(
        u_bat,
        horizon,
        action_per_hour,
        logpath,
        title="Battery actions",
        legends=["Battery Charge", "Battery Discharge"],
    )

    p.plot_control_actions(
        u_grid,
        horizon,
        action_per_hour,
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