import time
import logging
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from datetime import datetime, timedelta

import utils.plots as p
import utils.metrics as metrics
import utils.helpers as utils

from components.loads import Load
from ocp.scenario import ScenarioOCP
from components.PV import LinearPhotovoltaic
from components.battery import Battery


def scenario_mpc():
    """
    Main function for mpc-scheme with receding horizion.
    """

    np.random.seed(1)

    conf = utils.parse_config()
    testfile = conf["testfile"]
    trainfile = conf["trainfile"]

    logpath = None
    loggerpath = "./logs/all_logs.log"
    foldername = conf["foldername"]
    if foldername:
        logpath = utils.create_logs_folder(conf["logpath"], foldername)
        loggerpath = logpath + "logs.log"

    logging.basicConfig(
        filename=loggerpath,
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    perfect_predictions = conf["perfect_predictions"]

    actions_per_hour = conf["actions_per_hour"]
    horizon = conf["simulation_horizon"]

    T = conf["prediction_horizon"]
    N = conf["prediction_horizon"] * actions_per_hour
    N_scenarios = conf["N_scenarios"]

    simulation_horizon = horizon * actions_per_hour

    assert N_scenarios in [1, 3, 7, 9], "Only 1, 3, 7 or 9 branches allowed"

    # Get data
    observations = pd.read_csv(testfile, parse_dates=["date"]).fillna(0)
    observations = observations[
        observations["date"]
        >= datetime(conf["start_year"], conf["start_month"], conf["start_day"])
    ]

    assert (
        simulation_horizon + N < observations.shape[0]
    ), "Not enough data for simulation"

    # Read forecast file
    solcast_forecasts = pd.read_csv(
        conf["solcast_file"], parse_dates=["time", "collected"]
    ).fillna(0)

    # Read weighting files
    load_weights = pd.read_csv("./data/load_weights.csv")
    pv_weights = pd.read_csv("./data/pv_weights.csv")

    current_time = observations.date.iloc[0]

    forecast = solcast_forecasts[
        solcast_forecasts["collected"] == current_time - timedelta(minutes=60)
    ]

    obs = observations[
        (observations["date"] >= current_time)
        & (observations["date"] <= current_time + timedelta(minutes=10 * N))
    ]
    # Initialize components
    E = pd.read_csv(conf["price_file"], parse_dates=["time"])
    L = Load(N, testfile, "L", current_time)
    B = Battery(T, N, **conf["battery"])
    PV = LinearPhotovoltaic(trainfile)
    sys_metrics = metrics.SystemMetrics()

    # Define variables
    Pbc = []
    Pbd = []
    Pgs = []
    Pgb = []

    Pgb_p = 0
    Pgb_p_all = [Pgb_p]

    primary_Pgb = 0
    primary_Pgs = 0
    prob = [1]

    pv_measured = []
    l_measured = []
    errors = []
    E_measured = []
    time_stamps = [current_time]

    prediction_time = 0
    solver_time = 0
    filter_ = [str(i) for i in range(N)]

    # Initialize and build optimal control problem
    ocp = ScenarioOCP(T, N, N_scenarios)
    s_data = ocp.s_data(0)

    s0, lbs, ubs, lbg, ubg = ocp.build_scenario_ocp()

    # Initialize plots and progressbar
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    bar = progressbar.ProgressBar(
        maxval=simulation_horizon,
        widgets=[
            progressbar.Bar("=", "[", "]"),
            " Finished with",
            progressbar.Percentage(),
        ],
    )
    bar.start()

    # MPC loop
    for step in range(simulation_horizon):

        step_start = time.time()

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

            l_prediction, l_lower, l_upper = L.get_statistic_scenarios(
                current_time, step
            )
            l_prediction = L.linear_mixture(l_prediction, l_true)

            prediction_time += time.time() - pred_time

        if N_scenarios == 1:
            pv_scenarios = [pv_prediction]
            l_scenarios = [l_prediction]

        else:

            # Construct scenarios and weights
            l_lower, l_upper = L.get_minmax_day(current_time, step)
            l_probs = np.asarray(
                [
                    utils.get_probability_from_file(load_weights, step, i, filter_)
                    for i in range(3)
                ]
            )

            pv_probs = np.asarray(
                [
                    utils.get_probability_from_file(pv_weights, step, i, filter_)
                    for i in range(3)
                ]
            )

            pv_upper = PV.predict(ref.temp.values, ref.GHI90.values)
            pv_lower = PV.predict(ref.temp.values, ref.GHI10.values)

            pv_scenarios = np.asarray([pv_upper, pv_prediction, pv_lower])
            l_scenarios = np.asarray([l_lower, l_prediction, l_upper])

            prob = np.multiply(pv_probs, l_probs)
            prob /= np.sum(prob)  # Scale to one

            if N_scenarios == 7:
                pv_scenarios = [
                    pv_upper,
                    pv_upper,
                    pv_prediction,
                    pv_prediction,
                    pv_prediction,
                    pv_lower,
                    pv_lower,
                ]
                l_scenarios = [
                    l_lower,
                    l_prediction,
                    l_lower,
                    l_prediction,
                    l_upper,
                    l_prediction,
                    l_upper,
                ]
                prob = np.multiply(
                    np.asarray(
                        [
                            pv_probs[0],
                            pv_probs[0],
                            pv_probs[1],
                            pv_probs[1],
                            pv_probs[1],
                            pv_probs[2],
                            pv_probs[2],
                        ]
                    ),
                    np.asarray(
                        [
                            l_probs[0],
                            l_probs[1],
                            l_probs[0],
                            l_probs[1],
                            l_probs[2],
                            l_probs[1],
                            l_probs[2],
                        ]
                    ),
                )

            elif N_scenarios == 9:
                pv_scenarios = np.repeat(pv_scenarios, 3, axis=0)
                l_scenarios = np.tile(l_scenarios, (3, 1))

                prob = np.multiply(np.repeat(pv_probs, 3), np.tile(l_probs, 3))
                prob /= np.sum(prob)

        # Plot scenarios and predictions
        if N_scenarios <= 3 and step % 18 == 0:
            colors = [i for i in get_cmap("tab10").colors]
            t_1 = [current_time + timedelta(minutes=10 * i) for i in range(N)]
            if step == 0:
                label_s = ["Upper", "Prediction", "Lower"]
                label_obs = "Observation"
            else:
                label_s = [None] * 3
                label_obs = None
            for i in range(len(pv_scenarios)):
                ax1.plot(
                    t_1,
                    pv_scenarios[i],
                    label=label_s[1],
                    color=colors[1],
                )
                ax2.plot(
                    t_1,
                    l_scenarios[i],
                    label=label_s[1],
                    color=colors[1],
                )

            ax1.plot(t_1, obs.PV[1:], label=label_obs, color=colors[0])
            ax2.plot(t_1, obs.L[1:], label=label_obs, color=colors[0])

        # Update parameters
        for i in range(N_scenarios):
            s0["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC()
            lbs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC()
            ubs["scenario" + str(i), "states", 0, "SOC"] = B.get_SOC()

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

        # Solve OCP
        sol_time = time.time()
        xk_opt, Uk_opt = ocp.solve_nlp(
            [s0, lbs, ubs, lbg, ubg], s_data, np.argmax(prob)
        )
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
        time_stamps.append(current_time)

        Uk_temp = np.copy(Uk_opt)

        e, uk = utils.primary_controller(
            B.get_SOC(), Uk_opt, obs["PV"].values[0], obs["L"].values[0]
        )

        Pgb_p = np.max([Pgb_p_all[-1], uk[2]])
        Pgb_p_all.append(Pgb_p)

        errors.append(e)

        Pbc.append(uk[0])
        Pbd.append(uk[1])
        Pgb.append(uk[2])
        Pgs.append(uk[3])

        # Calculate price from primary controller
        temp_sale_diff = uk[3] - Uk_temp[3]
        temp_buy_diff = uk[2] - Uk_temp[2]

        if temp_buy_diff >= 0:
            primary_Pgb = np.abs(
                (
                    (uk[2] - Uk_temp[2])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )
            primary_Pgs = 0

        if temp_sale_diff >= 0:
            primary_Pgs = np.abs(
                (
                    (uk[3] - Uk_temp[3])
                    * E[E.time == current_time].price.values[0]
                    / actions_per_hour
                )
            )
            primary_Pgb = 0

        B.simulate_SOC(xk_opt, [uk[0], uk[1]])

        step_time = time.time() - step_start

        sys_metrics.update_metrics(
            [Pbc[-1], Pbd[-1], Uk_temp[2], Uk_temp[3]],
            E[E.time == current_time].price.values[0] / actions_per_hour,
            e,
            Pgb_p_all[-1],
            primary_Pgb,
            primary_Pgs,
            step_time,
        )

        bar.update(step)

    sys_metrics.print_metrics(B.x_sim, E_measured)

    df = utils.create_datafile(
        [
            time_stamps[1:],
            100 * np.asarray(B.x_sim[1:]),
            100 * np.asarray(B.x_opt[1:]),
            np.asarray(Pbc) - np.asarray(Pbd),
            np.asarray(Pgb) - np.asarray(Pgs),
            ocp.Pbc - ocp.Pbd,
            ocp.Pgb - ocp.Pgs,
            np.asarray(errors),
            pv_measured,
            l_measured,
            E_measured,
            Pgb_p_all[1:],
        ],
        [
            "date",
            "SOC_sim",
            "SOC_opt",
            "Pb_sim",
            "Pg_sim",
            "Pb_opt",
            "Pg_opt",
            "Errors",
            "PV",
            "Load",
            "Spot_prices",
            "P_peak",
        ],
        logpath=logpath,
    ).set_index("date")

    # Plotting
    u = np.asarray(
        [np.asarray(Pbc) - np.asarray(Pbd), np.asarray(Pgb) - np.asarray(Pgs)]
    )
    p.plot_from_df(
        df,
        ["Pb_sim"],
        title="Battery Action",
        upsample=True,
        legends=["Battery"],
        logpath=logpath,
    )

    p.plot_from_df(
        df,
        ["Pg_sim", "Pg_opt", "P_peak"],
        title="Grid Control Actions",
        upsample=True,
        legends=["Primary Control", "Optimal", "Peak Power"],
        logpath=logpath,
    )
    p.plot_from_df(
        df,
        ["SOC_sim"],
        ylabel="SOC [%]",
        title="State of Charge",
        logpath=logpath,
    )
    p.plot_from_df(
        df,
        ["PV", "Load"],
        title="Measured PV and Load",
        upsample=True,
        legends=["PV", "Load"],
        logpath=logpath,
    )

    p.plot_from_df(
        df,
        ["Spot_prices"],
        title="Spot Prices",
        ylabel="Price [NOK]",
        upsample=True,
        logpath=logpath,
    )

    p.format_figure(
        fig1,
        ax1,
        df.index,
        title="PV Scenarios",
        logpath=logpath,
    )
    p.format_figure(
        fig1,
        ax2,
        df.index,
        title="Load Scenarios",
        logpath=logpath,
    )

    if conf["plot"]:
        fig1.tight_layout()
        if logpath:
            fig1.savefig(logpath + "scenarios" + ".pdf", format="pdf")
        plt.show(block=True)


if __name__ == "__main__":
    scenario_mpc()
