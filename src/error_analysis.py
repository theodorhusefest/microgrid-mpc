import time
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime, timedelta

from components.loads import Load
from components.PV import Photovoltaic, LinearPhotovoltaic
from statsmodels.stats.diagnostic import lilliefors
from sklearn.metrics import mean_squared_error

sn.set_theme(
    context="paper",
    style="white",
    font_scale=1.8,
    rc={"lines.linewidth": 2},
    palette="tab10",
)


def load_analysis(N, L, data, plot=True):
    error_df = pd.DataFrame()

    rmse = []
    days = 0
    time_skipped = 144 * days
    all_errors = pd.DataFrame(columns=["prediction", "lower", "upper"])

    for i in range(time_skipped, data.shape[0] - N):
        current_time = data.date[i]
        L_true = data[
            (data["date"] > current_time)
            & (data["date"] <= (current_time + timedelta(minutes=10 * N)))
        ][L.column].values

        # L_pred = L.get_previous_day(current_time, measurement=data.L.iloc[i], days=days)
        L_pred, _, _ = L.get_statistic_scenarios(current_time, i)

        L_lower, L_upper = L.get_minmax_day(current_time, i)
        # L_pred = L.interpolate_prediction(L_pred, data.L.iloc[i])
        error = L_true - L_pred
        error_df[i - time_skipped] = error

        temp = pd.DataFrame(
            data={
                "prediction": L_true - L_pred,
                "lower": L_true - L_lower,
                "upper": L_true - L_upper,
            }
        )

        all_errors = all_errors.append(temp)

        if error.mean() >= 0:
            rmse.append(np.sqrt(mean_squared_error(L_true, L_pred)))
        else:
            rmse.append(-np.sqrt(mean_squared_error(L_true, L_pred)))

    error_df = error_df.transpose()

    if plot:
        print("****** Load statistics ******")
        print(error_df.describe())
        print("\n" + "*" * 30 + " Lilliefors " + "*" * 30)
        print(
            "Lilliefors test-statistic:",
            lilliefors(error_df.values.flatten(), dist="norm")[1],
        )
        print("RMSE total: ", np.sqrt(np.power(error_df.values, 2).mean()))
        plot_boxplot(error_df, "Load")
        # plot_daily_errors(error_df.values, "PV")
        plot_error_hist(error_df.values, "Load")
        plt.show()

    # all_errors.to_csv("./data/scenario_errors_load.csv")
    error_df["date"] = data.date.iloc[time_skipped : data.shape[0] - N].values
    error_df["rmse"] = rmse
    return error_df


def pv_analysis(N, pv, data, forecasts, plot=True):
    """
    Calculates the error between prediction and observed production
    """
    error_df = pd.DataFrame()
    forecast = forecasts[
        forecasts["collected"] == data.date.iloc[0] - timedelta(minutes=60)
    ]
    rmse = []

    all_errors = pd.DataFrame(columns=["prediction", "lower", "upper"])

    # mean_day = np.append(pv.get_mean_day("airTemp"), pv.get_mean_day("airTemp"))
    prediction_time = 0
    num_preds = 0
    sim_horizon = int((data.shape[0] - N))

    for i in range(sim_horizon):
        current_time = data.date[i]
        if current_time.minute == 30:
            new_forecast = forecasts[
                forecasts["collected"] == current_time - timedelta(minutes=30)
            ]
            if new_forecast.empty:
                print("Could not find forecast, using old forecast")
            else:
                forecast = new_forecast

        df = (
            pd.merge(
                forecast,
                data,
                left_on="time",
                right_on="date",
                suffixes=("_forecast", "_obs"),
            )
            .fillna(0)
            .iloc[: N + 1]
        )
        # df = df.set_index("time")
        measurement = data[data.date == current_time]["PV"].iloc[0]
        # df["PV_pred"] = pv.predict(
        #   df.temp.values, df.GHI_forecast.values, measurement
        # )
        df = df.iloc[1:]

        pred_start = time.time()
        df["PV_pred"] = pv.predict(df.temp.values, df.GHI_forecast.values, None)
        prediction_time += time.time() - pred_start
        num_preds += 1
        error = df["PV"] - df["PV_pred"]

        temp = pd.DataFrame(
            data={
                "prediction": df["PV"] - df["PV_pred"],
                "lower": df["PV"] - pv.predict(df.temp.values, df.GHI10.values),
                "upper": df["PV"] - pv.predict(df.temp.values, df.GHI90.values),
            }
        )

        all_errors = all_errors.append(temp)

        if True and i == 232:
            plt.figure(figsize=(10, 5))

            plt.plot(df["PV"], label="PV True")
            plt.plot(df["PV_pred"], label="Prediction With Measurement")
            plt.plot(
                pv.predict(df.temp.values, df.GHI_forecast.values),
                label="Prediction Without Measurement",
            )
            plt.legend()
            plt.xlabel("Prediction step")
            plt.ylabel("Power [kW]")
            plt.title("Prediction with Linear Mixture")
            plt.tight_layout()
            plt.savefig("../figs/linear_mixture_pred.png", format="png")
            plt.show()

        if error.mean() >= 0:
            rmse.append(np.sqrt(mean_squared_error(df["PV"], df["PV_pred"])))
        else:
            rmse.append(-np.sqrt(mean_squared_error(df["PV"], df["PV_pred"])))

        error_df[i - 1] = error

    # all_errors.to_csv("./data/scenario_errors_pv.csv")
    error_df = error_df.transpose()

    if plot:
        print("****** PV statistics ******")
        print(error_df.describe())
        print("Average time per forecast {}".format(prediction_time / num_preds))
        print("\n" + "*" * 30 + " Lilliefors " + "*" * 30)
        print(
            "Lilliefors test-statistic:",
            lilliefors(error_df.values.flatten(), dist="norm")[1],
        )
        print("RMSE total: ", np.sqrt(np.power(error_df.values, 2).mean()))

        plot_boxplot(error_df, "PV")
        # plot_daily_errors(error_df.values, "PV")
        plot_error_hist(error_df.values, "PV")
        plt.show()
    error_df["date"] = data.date
    error_df["rmse"] = rmse
    return error_df


def plot_boxplot(df, name):
    """
    Plots a boxplot of the prediction errors at each timestep
    """
    ax = df.plot.box(figsize=(10, 5))
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Error")
    ax.set_title("{} - Error at Prediction Step".format(name))


def plot_error_hist(errors, name):
    """
    Plots a histrogram of all errors
    """
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=100)
    plt.title("{} - Error distribution".format(name))


def plot_daily_errors(daily_errors, name):
    """
    Plots errors at the correct timesteps
    """
    daily_mean = daily_errors.mean(axis=0)
    plt.figure(figsize=(10, 5))
    for i in range(daily_mean.shape[0]):
        plt.scatter(range(i, i + daily_mean.shape[1]), daily_mean[i])
    plt.title("{} - Average prediction errors at timestep".format(name))
    plt.xlabel("Timestep")
    plt.ylabel("Error")


def estimate_errors(N, PV, train_file, test_file, forecast_file, stopdate=None):
    """
    Calculates and saves load and pv errors to a csv.
    """

    if not stopdate:
        stopdate = datetime(2100, 12, 30)

    test_data = pd.read_csv(test_file, parse_dates=["date"])

    current_time = test_data.date[0]

    L = Load(N, train_file, "L", current_time)
    load_errors = load_analysis(N, L, test_data, plot=True)

    observations = pd.read_csv(test_file, parse_dates=["date"]).fillna(0)
    solcast_forecasts = pd.read_csv(
        forecast_file, parse_dates=["time", "collected"]
    ).fillna(0)

    pv_errors = pv_analysis(
        N,
        PV,
        observations,
        solcast_forecasts,
        plot=True,
    )

    # pv_errors = pv_errors.loc[~(pv_errors == 0).all(axis=1)]
    # load_errors = load_errors.loc[~(load_errors == 0).all(axis=1)]

    pv_errors.to_csv("./data/pv_errors_date.csv")
    load_errors.to_csv("./data/load_errors_date.csv")

    return pv_errors, load_errors


if __name__ == "__main__":
    train_file = "./data/23.4_train.csv"
    test_file = "./data/23.4_test.csv"
    test_data_load = "./data/23.4_test.csv"
    pv = LinearPhotovoltaic(train_file)
    # pv = Photovoltaic()
    estimate_errors(
        60,
        pv,
        test_file,
        train_file,
        "./data/solcast_cleaned.csv",
    )
