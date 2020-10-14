import time
from casadi import *
import matplotlib.pyplot as plt
from utils.plots import plot_SOC, plot_control_actions

from open_loop import open_loop_optimization
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load


def run_mpc():
    """
    Main function for mpc-scheme with receding horizion.

    Assumptions:
    - Only get weather predicitons at start of interval

    """
    DAYS = 1
    ACTION_PER_HOUR = 6
    TIMEHORIZON = DAYS * 24  # timehorizon

    start = time.time()
    step_time = start

    x_inital = 0.5
    x_noise = True
    constant_offset = 0

    opts_open_loop = {
        "C_MAX": 700,
        "nb_c": 0.8,
        "nb_d": 0.8,
        "x_min": 0.3,
        "x_max": 0.9,
        "x_ref": 0.7,
        "Pb_max": 1000,
        "Pg_max": 500,
        "battery_cost": 1,
        "grid_buy": 1,
        "grid_sell": 1,
        "ref_cost": 1,
        "verbose": False,
    }

    # Get predicted for the time period
    pv_predictied = simulate_pv_cell(
        samples_per_hour=ACTION_PER_HOUR,
        max_power=400,
        days=DAYS,
        plot=True,
        add_noise=False,
    )
    pl_predictied = simulate_p_load(
        samples_per_hour=ACTION_PER_HOUR,
        max_power=2000,
        days=DAYS,
        plot=False,
        add_noise=False,
    )

    x = np.asarray([x_inital])
    u0 = np.asarray([])
    u1 = np.asarray([])
    u2 = np.asarray([])
    u3 = np.asarray([])

    xk = x_inital
    for H in range(TIMEHORIZON):
        T = TIMEHORIZON - H
        N = T * ACTION_PER_HOUR
        x_opt, U_opt = open_loop_optimization(
            xk,
            T,
            N,
            pv_predictied[H * ACTION_PER_HOUR : :],
            pl_predictied[H * ACTION_PER_HOUR : :],
            **opts_open_loop,
            plot=False
        )

        xk = x_opt[ACTION_PER_HOUR] - constant_offset  # Fake/no error measurement

        if x_noise and np.random.randint(0, 2):
            xk = xk + np.random.normal(0, 0.05)

        # Get the next control actions
        uk = [u[0:ACTION_PER_HOUR] for u in U_opt]

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

    plot_control_actions(u, TIMEHORIZON, ACTION_PER_HOUR)

    plot_control_actions(
        u_bat,
        TIMEHORIZON,
        ACTION_PER_HOUR,
        title="Battery actions",
        legends=["Battery Charge", "Battery Discharge"],
    )

    plot_control_actions(
        u_grid,
        TIMEHORIZON,
        ACTION_PER_HOUR,
        title="Grid actions",
        legends=["Grid Buy", "Grid Sell"],
    )

    plot_SOC(x, TIMEHORIZON)

    stop = time.time()
    print("\nFinished optimation on {}s".format(np.around(stop - start, 2)))

    plt.show()


run_mpc()