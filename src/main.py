import time
from ast import literal_eval
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
    observations = observations[observations["date"] >= datetime(2021, 4, 7)]
    solcast_forecasts = pd.read_csv(
        conf["solcast_file"], parse_dates=["time", "collected"]
    ).fillna(0)
    load_scenario_file = pd.read_csv("./data/load_scenarios.csv")
    pv_scenario_file = pd.read_csv("./data/pv_scenarios.csv")

    current_time = observations.date.iloc[0]
    print("Starting simulation at ", current_time)

    forecast = solcast_forecasts[
        solcast_forecasts["collected"] == current_time - timedelta(minutes=60)
    ]

    obs = observations[
        (observations["date"] >= current_time)
        & (observations["date"] <= current_time + timedelta(minutes=10 * N))
    ]

    E = pd.read_csv(conf["price_file"], parse_dates=["time"])

    l = Load(N, testfile, "L", groundtruth=trainfile)
    B = Battery(T, N, **conf["battery"])
    PV = LinearPhotovoltaic(trainfile)

    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    Pgb_p_all = []
    Pgb_p = 1

    Primary_Pgb = 0
    Primary_Pgb_undelivered = 0
    Primary_Pgs = 0
    Primary_Pgs_undelivered = 0

    pv_measured = []
    l_measured = []
    errors = []
    E_measured = []
    c_violations = 0

    prediction_time = 0
    simulation_time = 0
    reduction_time = 0
    solver_time = 0

    # Initilize Montecarlo
    N_sim = 100
    # monte_carlo = njit()(monte_carlo_simulations)
    load_errors = pd.read_csv("./data/load_errors.csv").drop(["Unnamed: 0"], axis=1)
    load_errors = load_errors[~np.isnan(load_errors).any(axis=1)]
    load_errors = shuffle_dataframe(load_errors).values
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

    filter_ = [str(i) for i in range(N)]
    prob = [1, 1, 1]
    for step in range(simulation_horizon - N):

        # Get measurements
        pv_true = obs["PV"].values[0]
        l_true = obs["L"].values[0]

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

        E_prediction = E[
            (E.time > current_time)
            & (E.time <= current_time + timedelta(minutes=10 * (N)))
        ].price.values

        # Get predictions
        if perfect_predictions:
            pv_prediction = obs["PV"].values[1:]
            l_prediction = obs["L"].values[1:]
        else:
            pred_time = time.time()
            pv_prediction = PV.predict(
                ref.temp.values, ref.GHI.values, obs["PV"].iloc[0]
            )
            l_prediction = l.get_previous_day(
                current_time, measurement=None, days=1
            )  # l_true)
            prediction_time += time.time() - pred_time

        if N_scenarios == 1:
            pv_scenarios = [pv_prediction]
            l_scenarios = [l_prediction]

        elif True:
            pv_upper = PV.predict(ref.temp.values, ref.GHI90.values, obs["PV"].iloc[0])
            pv_lower = PV.predict(ref.temp.values, ref.GHI10.values, obs["PV"].iloc[0])

            # l_sims = monte_carlo(N, N_sim, l_prediction, load_errors)
            # l_min = np.min(l_sims, axis=0)
            # l_max = np.max(l_sims, axis=0)

            l_min, l_max, l_mean = [
                utils.get_scenario_from_file(load_scenario_file, step, i, filter_)
                for i in range(3)
            ]

            l_probs = np.asarray(
                [
                    utils.get_probability_from_file(
                        load_scenario_file, step, i, filter_
                    )
                    for i in range(3)
                ]
            )

            pv_probs = np.asarray(
                [
                    utils.get_probability_from_file(pv_scenario_file, step, i, filter_)
                    for i in range(3)
                ][::-1]
            )

            l_lower = l.interpolate_prediction(l_prediction - l_min, l_true)
            l_upper = l.interpolate_prediction(l_prediction - l_max, l_true)

            pv_scenarios = np.asarray([pv_upper, pv_prediction, pv_lower])
            l_scenarios = np.asarray([l_lower, l_prediction, l_upper])

            prob = np.multiply(pv_probs, l_probs)
            prob /= np.sum(prob)  # Scale to one

            if N_scenarios == 9:

                pv_scenarios = np.repeat(pv_scenarios, 3, axis=0)
                l_scenarios = np.tile(l_scenarios, (3, 1))

                prob = np.multiply(np.repeat(pv_probs, 3), np.tile(l_probs, 3))
                prob /= np.sum(prob)

        if step % 20 == 0:
            for i in range(len(pv_scenarios)):
                ax1.plot(range(step, step + N), pv_scenarios[i], color="red")
                ax2.plot(range(step, step + N), l_scenarios[i], color="red")

            ax1.plot(range(step, step + N), pv_prediction, color="blue")
            ax1.plot(range(step, step + N), obs.PV[1:], color="green")
            ax2.plot(range(step, step + N), l_prediction, color="blue")
            ax2.plot(range(step, step + N), obs.L[1:], color="green")

        # Update parameters
        for i in range(N_scenarios):
            "scenario" + str(i)
            s0["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            lbs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)
            ubs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC(openloop)

            s0["scenario" + str(i), "states", 0, "Pgb_p"] = Pgb_p
            lbs["scenario" + str(i), "states", 0, "Pgb_p"] = Pgb_p
            ubs["scenario" + str(i), "states", 0, "Pgb_p"] = Pgb_p

            for k in range(N):
                s_data["scenario" + str(i), "data", k, "pv"] = pv_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "l"] = l_scenarios[i][k]
                s_data["scenario" + str(i), "data", k, "E"] = (
                    E_prediction[k] / actions_per_hour
                )
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

        pv_measured.append(pv_true)
        l_measured.append(l_true)
        E_measured.append(E_prediction[0])
        Uk_temp = np.copy(Uk_opt)

        e, uk = utils.primary_controller(
            xk_opt, Uk_opt, obs["PV"].values[0], obs["L"].values[0]
        )

        errors.append(e)

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        temp_sale_diff = uk[3] - Uk_temp[3]
        temp_buy_diff = uk[2] - Uk_temp[2]

        if temp_buy_diff >= 0:
            Primary_Pgb += np.abs(
                (
                    (uk[2] - Uk_temp[2])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )
        else:
            Primary_Pgb_undelivered += np.abs(
                (
                    (uk[2] - Uk_temp[2])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )

        if temp_sale_diff >= 0:
            Primary_Pgs += np.abs(
                (
                    (uk[3] - Uk_temp[3])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )
        else:
            Primary_Pgs_undelivered += np.abs(
                (
                    (uk[3] - Uk_temp[3])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        if B.get_SOC(openloop) < 0.18 or B.get_SOC(openloop) > 0.82:
            c_violations += 1

        sys_metrics.update_metrics(
            [Pbc[-1], Pbd[-1], Uk_temp[2], Uk_temp[3]],
            E[E.time == current_time].price.values[0] / actions_per_hour,
            e,
        )

        utils.print_status(
            step, [B.get_SOC(openloop)], step_time, every=int(simulation_horizon / 3)
        )
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

    p.plot_data(
        np.asarray([E_measured]),
        title="Spot Prices",
        legends=["Spotprice"],
        logpath=logpath,
    )

    p.plot_data(
        np.asarray([Pgb_p_all]),
        title="Peak Power",
        legends=["Peak Power"],
        logpath=logpath,
    )

    stop = time.time()
    print("Primary controller extra bought for {}".format(Primary_Pgb))
    print("Primary controller undelivered buy for {}".format(Primary_Pgb_undelivered))
    print("Primary controller extra sold for {}".format(Primary_Pgs))
    print("Primary controller undelivered sale for {}".format(Primary_Pgs_undelivered))
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
