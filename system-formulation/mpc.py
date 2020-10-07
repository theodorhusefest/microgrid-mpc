import time
from casadi import *
import matplotlib.pyplot as plt

from open_loop import open_loop_optimization

from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load

DAYS = 1
ACTION_PER_HOUR = 5
TIMEHORIZON = DAYS * 24  # timehorizon
MAX_N = TIMEHORIZON * ACTION_PER_HOUR # Maximum prediction horizion

def run_mpc():
    """
    Main function for mpc-scheme with receding horizion.

    Assumptions:
    - Only get weather predicitons at start of interval

    """
    start = time.time()
    step_time = start

    x_inital = 0.8
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
        "battery_cost": 0.1,
        "grid_buy": 10000,
        "grid_sell": 1000,
        "ref_cost": 0.1,
        "verbose": False

    }
    
    # Get predicted for the time period
    pv_predictied = simulate_pv_cell(
        samples_per_hour=ACTION_PER_HOUR,
        max_power=300,
        days=DAYS,
        plot=True,
        add_noise=False)
    pl_predictied = simulate_p_load(
        samples_per_hour=ACTION_PER_HOUR,
        max_power=300,
        days=DAYS,
        plot=False, 
        add_noise=False)


    x = [x_inital]
    u0 = []
    u1 = []
    u2 = []
    u3 = []

    xk = x_inital
    for H in range(TIMEHORIZON):
        T = TIMEHORIZON - H
        N = T*ACTION_PER_HOUR
        x_opt, U_opt = open_loop_optimization(
            xk,
            T,
            N,
            pv_predictied[H*ACTION_PER_HOUR::], 
            pl_predictied[H*ACTION_PER_HOUR::],
            **opts_open_loop, plot = False)

        xk = x_opt[ACTION_PER_HOUR] - constant_offset # Fake/no error measurement

        if x_noise and np.random.randint(0,2):
            xk = xk + np.random.normal(0, 0.01)


        # Get the next control actions
        uk = [ u[0:ACTION_PER_HOUR] for u in U_opt ]
        x.append(xk)
        u0.append(uk[0])
        u1.append(uk[1])
        u2.append(uk[2])
        u3.append(uk[3])

        print("\nFinshed iteration hour {}. Current step took {}s".format(
            H, np.around(time.time()-step_time, 2)))
        step_time = time.time()

    u0 = np.asarray(u0).flatten()
    u1 = np.asarray(u1).flatten()
    u2 = np.asarray(u2).flatten()
    u3 = np.asarray(u3).flatten()

    # Plotting

    u0_plot = np.repeat(u0, int(60/ACTION_PER_HOUR))
    u1_plot = np.repeat(u1, int(60/ACTION_PER_HOUR))
    u2_plot = np.repeat(u2, int(60/ACTION_PER_HOUR))
    u3_plot = np.repeat(u3, int(60/ACTION_PER_HOUR))
    t = np.asarray(range(TIMEHORIZON*60))

    plt.figure()
    plt.plot(t, u0_plot)
    plt.plot(t, -u1_plot)
    plt.plot(t, u2_plot)
    plt.plot(t, -u3_plot)
    plt.xlabel('Time [h]')
    plt.xticks(np.linspace(0, (TIMEHORIZON*60), TIMEHORIZON), range(TIMEHORIZON))
    plt.ylabel('Power [kW]')
    plt.title('All controls')
    plt.legend(['Pb_charge','Pb_discharge', 'Pg_buy', 'Pg_sell'])

    plt.figure()
    plt.plot(t, u0_plot)
    plt.plot(t, -u1_plot)
    plt.xlabel('Time [h]')
    plt.xticks(np.linspace(0, (TIMEHORIZON*60), TIMEHORIZON), range(TIMEHORIZON))
    plt.ylabel('Power [kW]')
    plt.title('Battery controls')
    plt.legend(['Pb_charge','Pb_discharge'])

    plt.figure()
    plt.plot(t, u2_plot)
    plt.plot(t, -u3_plot)
    plt.xlabel('Time [h]')
    plt.xticks(np.linspace(0, (TIMEHORIZON*60), TIMEHORIZON), range(TIMEHORIZON))
    plt.ylabel('Power [kW]')
    plt.title('Battery controls')
    plt.legend(['Pg_buy', 'Pg_sell'])

    plt.figure()
    plt.plot(range(TIMEHORIZON+1), x)
    plt.xlabel('Time [h]')
    plt.ylabel('SOC [%]')
    plt.title('State of charge')
    plt.ylim([0.25, 0.95])

    stop = time.time()
    print('\nFinished optimation on {}s'.format(np.around(stop-start, 2)))

    plt.show()


run_mpc()