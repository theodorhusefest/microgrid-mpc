import time
from casadi import *
import matplotlib.pyplot as plt
from utils.plots import plot_SOC, plot_control_actions
from utils.helpers import create_logs_folder, parse_config

from open_loop import open_loop_optimization
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load


def main():
    """
    Main function for mpc-scheme with receding horizion.

    Assumptions:
    - Only get weather predicitons at start of interval

    """
    conf = parse_config()

    logpath = create_logs_folder(conf["logpath"])
    days = conf["days"]
    action_per_hour = conf["action_per_hour"]
    TIMEHORIZON = days * 24  # timehorizon

    start = time.time()
    step_time = start

    x_inital = conf["x_inital"]
    x_noise = conf["x_noise"]

    # Get predicted for the time period
    pv_conf = conf["pv-cell"]
    pv_predictied = simulate_pv_cell(
        samples_per_hour=action_per_hour,
        max_power=pv_conf["max_power"],
        days=days,
        plot=True,
        logpath=logpath,
        add_noise=pv_conf["add_noise"],
    )
    pl_conf = conf["p-load"]
    pl_predictied = simulate_p_load(
        samples_per_hour=action_per_hour,
        max_power=pl_conf["max_power"],
        days=days,
        plot=True,
        logpath=logpath,
        add_noise=pl_conf["add_noise"],
    )

    x = np.asarray([x_inital])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    xk = x_inital
    for H in range(TIMEHORIZON):
        T = TIMEHORIZON - H
        N = T * action_per_hour
        x_opt, U_opt = open_loop_optimization(
            xk,
            T,
            N,
            pv_predictied[H * action_per_hour : :],
            pl_predictied[H * action_per_hour : :],
            **conf["system"],
            plot=False
        )

        if conf["openloop"]:
            x = x_opt
            u0 = U_opt[0]
            u1 = U_opt[1]
            u2 = U_opt[2]
            u3 = U_opt[3]
            break

        xk = x_opt[action_per_hour]  # Fake/no error measurement

        if x_noise and np.random.randint(0, 2):
            xk += np.random.normal(0, x_noise)

        # Get the next control actions
        uk = [u[0:action_per_hour] for u in U_opt]

        x = np.append(x, xk)
        u0 = np.append(u0, uk[0])
        u1 = np.append(u1, uk[1])
        u2 = np.append(u2, uk[2])
        u3 = np.append(u3, uk[3])

        print(
            "\nFinshed iteration hour {}. Current step took {}s".format(
                H, np.around(time.time() - step_time, 2)
            )
        )
        step_time = time.time()

    # Plotting
    u = np.asarray([-u0, u1, u2, -u3])
    u_bat = np.asarray([-u0, u1])
    u_grid = np.asarray([u2, -u3])

    plot_control_actions(u, TIMEHORIZON, action_per_hour, logpath)

    plot_control_actions(
        u_bat,
        TIMEHORIZON,
        action_per_hour,
        logpath,
        title="Battery actions",
        legends=["Battery Charge", "Battery Discharge"],
    )

    plot_control_actions(
        u_grid,
        TIMEHORIZON,
        action_per_hour,
        logpath,
        title="Grid actions",
        legends=["Grid Buy", "Grid Sell"],
    )

    plot_SOC(x, TIMEHORIZON, logpath)

    stop = time.time()
    print("\nFinished optimation on {}s".format(np.around(stop - start, 2)))
    plt.show()


if __name__ == "__main__":
    main()