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
from components.PV import Photovoltaic
from ocp.nominel import NominelMPC
from components.loads import Load
from components.battery import Battery


def nominel_mpc():
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
    observations = pd.read_csv(datafile, parse_dates=["date"])
    # observations = observations[observations["date"] >= datetime(2021, 3, 11)]
    solcast_forecasts = pd.read_csv(
        conf["solcast_file"], parse_dates=["time", "collected"]
    )

    current_time = observations.date.iloc[0]

    forecast = solcast_forecasts[
        solcast_forecasts["collected"] == current_time - timedelta(minutes=60)
    ]

    obs = observations[observations["date"] == current_time]

    l = Load(N, loads_trainfile, "L", groundtruth=observations["L"])
    E = np.ones(2000)  # get_spot_price()
    B = Battery(T, N, **conf["battery"])
    PV = Photovoltaic()

    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    pv_measured = []
    l_measured = []
    errors = []

    ocp = NominelMPC(T, N)
    sys_metrics = metrics.SystemMetrics()

    x, lbx, ubx, lbg, ubg = ocp.build_nlp()

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
            (forecast["time"] >= current_time)
            & (forecast["time"] < current_time + timedelta(minutes=10 * (N + 1)))
        ]

        # Create predictions for next period
        if perfect_predictions:
            pv_ref = pv[step + 1 : step + N + 1]
            l1_ref = l1.perfect_pred(step)
            E_ref = E[step : step + N]
        else:
            pv_ref = PV.predict(ref.temp.values, ref.GHI.values)
            l_ref = l.scaled_mean_pred(l_true, step % 126)
            E_ref = E[step : step + N]

        forecasts = ocp.update_forecasts(pv_ref, l_ref, E_ref)

        xk_opt, Uk_opt = ocp.solve_nlp([x, lbx, ubx, lbg, ubg], forecasts)

        # Simulate the system after disturbances
        current_time += timedelta(minutes=10)
        obs = observations[observations["date"] == current_time]

        e, uk = utils.calculate_real_u(
            xk_opt, Uk_opt, obs["PV"].values[0], obs["L"].values[0]
        )
        errors.append(e)

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        sys_metrics.update_metrics(
            [Pbc[step], Pbd[step], Pgb[step], Pgs[step]], E[step], e
        )

        utils.print_status(step, [B.get_SOC(openloop)], step_time, every=50)
        step_time = time.time()

    sys_metrics.calculate_consumption_rate(Pgs, pv_measured)
    sys_metrics.calculate_dependency_rate(Pgb, l_measured)
    sys_metrics.print_metrics()

    # Plotting
    u = np.asarray(
        [np.asarray(Pbc) - np.asarray(Pbd), np.asarray(Pgb) - np.asarray(Pgs)]
    )
    if openloop:
        p.plot_control_actions(
            np.asarray([ocp.Pbc - ocp.Pbd, ocp.Pgb - ocp.Pgb]),
            horizon - T,
            actions_per_hour,
            logpath,
            legends=["Battery", "Grid"],
            title="Optimal Control Actions",
        )

    else:
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
    )

    p.plot_data([np.asarray(errors)], title="Errors")

    p.plot_data(
        np.asarray([pv_measured, l_measured]), title="PV & Load", legends=["PV", "L"]
    )

    # p.plot_data(np.asarray([E]), title="Spot Prices", legends=["Spotprice"])

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))

    plt.ion()
    if True:
        plt.show(block=True)


if __name__ == "__main__":
    nominel_mpc()
