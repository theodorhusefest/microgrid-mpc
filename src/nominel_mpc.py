import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from pprint import pprint
from datetime import datetime, timedelta

from components.spot_price import get_spot_price
from components.PV import Photovoltaic, LinearPhotovoltaic
from ocp.nominel import NominelMPC
from components.loads import Load
from components.battery import Battery


def nominel_mpc():
    """
    Main function for mpc-scheme with receding horizion.
    """
    np.random.seed(1)
    conf = utils.parse_config()
    testfile = conf["testfile"]
    trainfile = conf["trainfile"]

    logpath = None
    log = input("Log this run? ")

    if log in ["y", "yes", "Yes"]:
        foldername = input("Enter logfolder name? (enter to skip) ")
        logpath = utils.create_logs_folder(conf["logpath"], foldername)

    openloop = conf["openloop"]
    perfect_predictions = conf["perfect_predictions"]

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]
    simulation_horizon = horizon * actions_per_hour

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour

    start_time = time.time()
    step_time = start_time

    # Get data
    observations = pd.read_csv(testfile, parse_dates=["date"])
    observations = observations[observations["date"] >= datetime(2021, 4, 2)]
    solcast_forecasts = pd.read_csv(
        conf["solcast_file"], parse_dates=["time", "collected"]
    )

    current_time = observations.date.iloc[0]
    print("Starting simulation at ", current_time)
    forecast = solcast_forecasts[
        solcast_forecasts["collected"] == current_time - timedelta(minutes=60)
    ]

    obs = observations[
        (observations["date"] >= current_time)
        & (observations["date"] <= current_time + timedelta(minutes=10 * N))
    ]

    l = Load(N, trainfile, "L", groundtruth=observations["L"])
    E = np.ones(2000)  # get_spot_price()
    B = Battery(T, N, **conf["battery"])
    PV = LinearPhotovoltaic(trainfile)

    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    pv_measured = []
    l_measured = []
    errors = []

    c_violation = 0

    ocp = NominelMPC(T, N)
    sys_metrics = metrics.SystemMetrics()

    x, lbx, ubx, lbg, ubg = ocp.build_nlp()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for step in range(simulation_horizon - N):

        # Get measurements
        x["states", 0, "SOC"] = B.get_SOC(openloop)
        lbx["states", 0, "SOC"] = B.get_SOC(openloop)
        ubx["states", 0, "SOC"] = B.get_SOC(openloop)

        pv_true = obs["PV"].values[0]
        l_true = obs["L"].values[0]

        pv_measured.append(pv_true)
        l_measured.append(l_true)

        # Get new forecasts every hour
        if current_time.minute == 30:
            new_forecast = solcast_forecasts[
                solcast_forecasts["collected"] == current_time - timedelta(minutes=30)
            ]
            if new_forecast.empty:
                print("Could not find forecast, using old forecast")
            else:
                forecast = new_forecast

        ref = forecast[
            (forecast["time"] > current_time)
            & (forecast["time"] <= current_time + timedelta(minutes=10 * (N)))
        ]

        # Create predictions for next period
        if perfect_predictions:
            pv_ref = obs["PV"].values[1:]
            l_ref = obs["L"].values[1:]
            E_ref = E[step : step + N]
        else:
            pv_ref = PV.predict(ref.temp.values, ref.GHI.values)
            l_ref = l.scaled_mean_pred(l_true, step % 126)[1:]
            E_ref = E[step : step + N]

        forecasts = ocp.update_forecasts(pv_ref, l_ref, E_ref)

        xk_opt, Uk_opt = ocp.solve_nlp([x, lbx, ubx, lbg, ubg], forecasts, step)

        # Simulate the system after disturbances
        current_time += timedelta(minutes=10)
        obs = observations[
            (observations["date"] >= current_time)
            & (observations["date"] <= current_time + timedelta(minutes=10 * N))
        ]

        ax1.plot(range(step, step + N + 1), obs["L"], color="blue")
        ax1.plot(range(step + 1, step + N + 1), l_ref, color="red")
        ax2.plot(range(step, step + N + 1), obs["PV"], color="blue")
        ax2.plot(range(step + 1, step + N + 1), pv_ref, color="red")

        e, uk = utils.calculate_real_u(
            xk_opt, Uk_opt, obs["PV"].values[0], obs["L"].values[0]
        )
        errors.append(e)

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        if xk_opt < 0.3 or xk_opt > 0.9:
            print("SOC constraint violation")
            c_violation += 1

        sys_metrics.update_metrics(
            [Pbc[step], Pbd[step], Pgb[step], Pgs[step]], E[step], e
        )

        utils.print_status(step, [B.get_SOC(openloop)], step_time, every=50)
        step_time = time.time()

    sys_metrics.calculate_consumption_rate(Pgs, pv_measured)
    sys_metrics.calculate_dependency_rate(Pgb, l_measured)
    sys_metrics.print_metrics(B.get_SOC(openloop))
    print("Constraint violations:", c_violation)

    # Plotting
    u = np.asarray(
        [np.asarray(Pbc) - np.asarray(Pbd), np.asarray(Pgb) - np.asarray(Pgs)]
    )
    p.plot_control_actions(
        np.asarray([ocp.Pbc - ocp.Pbd, ocp.Pgb - ocp.Pgs]),
        horizon - T,
        actions_per_hour,
        logpath,
        legends=["Battery", "Grid"],
        title="Optimal Control Actions",
    )

    p.plot_control_actions(
        u,
        horizon - T,
        actions_per_hour,
        logpath,
        legends=["Battery", "Grid"],
        title="Simulated Control Actions",
    )

    p.plot_data(
        np.asarray([B.x_sim, B.x_opt]),
        title="State of charge",
        legends=["SOC", "SOC_opt"],
        logpath=logpath,
    )

    p.plot_data([np.asarray(errors)], title="Errors", logpath=logpath)

    p.plot_data(
        np.asarray([pv_measured, l_measured]),
        title="PV & Load",
        legends=["PV", "L"],
        logpath=logpath,
    )

    # p.plot_data(np.asarray([E]), title="Spot Prices", legends=["Spotprice"],logpath=logpath)

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    if logpath:
        fig1.savefig("{}-{}".format(logpath, "pv_forecast" + ".eps"), format="eps")
        fig2.savefig("{}-{}".format(logpath, "load_forecast" + ".eps"), format="eps")
    if True:
        plt.show(block=True)


if __name__ == "__main__":
    nominel_mpc()
