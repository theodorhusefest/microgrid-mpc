import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from datetime import datetime, timedelta

import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from components.loads import Load
from ocp.scenario import ScenarioOCP
from ocp.nominel import NominelMPC
from components.PV import Photovoltaic, LinearPhotovoltaic
from components.battery import Battery
from components.spot_price import get_spot_price
from utils.scenario_tree import build_scenario_tree, get_scenarios
from utils.monte_carlo import (
    monte_carlo_simulations,
    scenario_reduction,
    shuffle_dataframe,
)


def scenario_mpc():
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
    Nr = conf["robust_horizon"]
    branch_factor = conf["branch_factor"]
    N_scenarios = branch_factor ** Nr

    start_time = time.time()
    step_time = start_time

    # Get data
    observations = pd.read_csv(testfile, parse_dates=["date"]).fillna(0)
    observations = observations[observations["date"] >= datetime(2021, 3, 30)]
    solcast_forecasts = pd.read_csv(
        conf["solcast_file"], parse_dates=["time", "collected"]
    ).fillna(0)

    current_time = observations.date.iloc[0]
    print("Starting simulation at ", current_time)

    forecast = solcast_forecasts[
        solcast_forecasts["collected"] == current_time - timedelta(minutes=60)
    ]

    obs = observations[
        (observations["date"] >= current_time)
        & (observations["date"] <= current_time + timedelta(minutes=10 * N))
    ]

    l = Load(N, testfile, "L", groundtruth=trainfile)
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
    c_violations = 0

    prediction_time = 0
    simulation_time = 0
    reduction_time = 0
    solver_time = 0

    # Initilize Montecarlo
    N_sim = 3
    monte_carlo = njit()(monte_carlo_simulations)
    load_errors = pd.read_csv("./data/load_errors.csv").drop(["Unnamed: 0"], axis=1)
    load_errors = load_errors[~np.isnan(load_errors).any(axis=1)]
    load_errors = shuffle_dataframe(load_errors).values
    pv_errors = pd.read_csv("./data/pv_errors.csv").drop(["Unnamed: 0"], axis=1)
    pv_errors = pv_errors[~np.isnan(pv_errors).any(axis=1)]
    pv_errors = shuffle_dataframe(pv_errors).values

    l_min = np.min(load_errors, axis=0)
    l_max = np.max(load_errors, axis=0)

    # Build reference tree
    tree, leaf_nodes = build_scenario_tree(
        N, Nr, branch_factor, np.ones(N + 1), 0, np.ones(N + 1), 0
    )

    ocp = ScenarioOCP(T, N, N_scenarios)
    s_data = ocp.s_data(0)

    s0, lbs, ubs, lbg, ubg = ocp.build_scenario_ocp()

    sys_metrics = metrics.SystemMetrics()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    prob = [1, 1, 1]
    for step in range(simulation_horizon - N):

        # Get measurements
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

        # Get predictions
        if perfect_predictions:
            pv_prediction = obs["PV"].values[1:]
            l_prediction = obs["L"].values[1:]
        else:
            pred_time = time.time()
            pv_prediction = PV.predict(
                ref.temp.values, ref.GHI.values, obs["PV"].iloc[0]
            )
            # l_prediction = l.scaled_mean_pred(l_true, step % (144 - N))
            # l_prediction = l.get_prediction_mean(step % (144 - N))
            l_prediction = l.get_previous_day(current_time, l_true)

            prediction_time += time.time() - pred_time

        if N_scenarios == 1:
            pv_scenarios = [pv_prediction]
            l_scenarios = [l_prediction]

        elif True:
            pv_upper = PV.predict(ref.temp.values, ref.GHI90.values, obs["PV"].iloc[0])
            pv_lower = PV.predict(ref.temp.values, ref.GHI10.values, obs["PV"].iloc[0])
            pv_scenarios = [pv_upper, pv_prediction, pv_lower]

            # l_sims = monte_carlo_simulations(N, N_sim, l_prediction, load_errors)

            l_scenarios = [l_prediction + l_min, l_prediction, l_prediction + l_max]

            prob = [0.2, 0.6, 0.2]

        elif True:
            sim_time = time.time()
            pv_sims = np.clip(
                monte_carlo_simulations(N, N_sim, pv_prediction, pv_errors), 0, np.inf
            )
            l_sims = np.clip(
                monte_carlo_simulations(N, N_sim, l_prediction, load_errors), 0, np.inf
            )

            simulation_time += time.time() - sim_time

            red_time = time.time()
            # pv_scenarios = scenario_reduction(pv_sims, N, Nr, branch_factor)
            # l_scenarios = scenario_reduction(l_sims, N, Nr, branch_factor)[::-1]

            pv_scenarios = np.sort(pv_sims, axis=0)
            l_scenarios = np.sort(l_sims, axis=0)

            reduction_time += time.time() - red_time

        else:
            _, leaf_nodes = build_scenario_tree(
                N,
                Nr,
                branch_factor,
                np.append(np.asarray([obs["PV"].iloc[0]]), pv_prediction),
                25,
                np.append(np.asarray([obs["L"].iloc[0]]), l_prediction),
                25,
            )
            pv_scenarios = get_scenarios(leaf_nodes, "pv")
            l_scenarios = get_scenarios(leaf_nodes, "l")

        if step % N == 0:
            for i in range(len(pv_scenarios)):
                ax1.plot(range(step, step + N), pv_scenarios[i], color="red")
                ax2.plot(range(step, step + N), l_scenarios[i], color="red")

            ax1.plot(range(step, step + N), pv_prediction, color="blue")
            ax1.plot(range(step, step + N), obs.PV[1:], color="green")
            ax2.plot(range(step, step + N), l_prediction, color="blue")
            ax2.plot(range(step, step + N), obs.L[1:], color="green")

        # Update parameters
        for i in range(N_scenarios):
            s0["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            lbs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            ubs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)

            for k in range(N - 1):
                s_data["scenario" + str(i), "data", k, "pv"] = pv_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "l"] = l_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "E"] = 1
                s_data["scenario" + str(i), "data", k, "prob"] = prob[i]

        sol_time = time.time()
        xk_opt, Uk_opt = ocp.solve_nlp([s0, lbs, ubs, lbg, ubg], s_data)
        solver_time += time.time() - sol_time

        # Simulate the system after disturbances
        current_time += timedelta(minutes=10)

        obs = observations[
            (observations["date"] >= current_time)
            & (observations["date"] <= current_time + timedelta(minutes=10 * N))
        ]

        e, uk = utils.calculate_real_u(
            xk_opt, Uk_opt, obs["PV"].values[0], obs["L"].values[0]
        )

        errors.append(e)

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        if (
            B.get_SOC(openloop) < conf["system"]["x_min"]
            or B.get_SOC(openloop) > conf["system"]["x_max"]
        ):
            c_violations += 1

        sys_metrics.update_metrics([Pbc[-1], Pbd[-1], Pgb[-1], Pgs[-1]], E[-1], e)

        utils.print_status(step, [B.get_SOC(openloop)], step_time, every=50)
        step_time = time.time()

    sys_metrics.calculate_consumption_rate(Pgs, pv_measured)
    sys_metrics.calculate_dependency_rate(Pgb, l_measured)
    sys_metrics.print_metrics(B.get_SOC(openloop))
    print("Contraint violations", c_violations)

    ax1.set_title("PV")
    ax2.set_title("Load")

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
        title="PV and Load",
        legends=["PV", "Load"],
        logpath=logpath,
    )

    # p.plot_data(np.asarray([E]), title="Spot Prices", legends=["Spotprice"],logpath=logpath)

    stop = time.time()
    print("\nFinished optimation in {}s".format(np.around(stop - start_time, 2)))
    print("Prediction time was {}".format(np.around(prediction_time, 2)))
    print("Simulation time was {}".format(np.around(simulation_time, 2)))
    print("Reduction time was {}".format(np.around(reduction_time, 2)))
    print("Solver time was {}".format(np.around(solver_time, 2)))

    if logpath:
        fig1.savefig("{}-{}".format(logpath, "pv_scenarios" + ".eps"), format="eps")
        fig2.savefig("{}-{}".format(logpath, "load_scenarios" + ".eps"), format="eps")
    if True:
        plt.show(block=True)


if __name__ == "__main__":
    scenario_mpc()
