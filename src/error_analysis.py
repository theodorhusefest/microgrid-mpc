import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime, timedelta

from components.loads import Load
from components.PV import Photovoltaic, LinearPhotovoltaic
from statsmodels.stats.diagnostic import lilliefors
from sklearn.metrics import mean_squared_error


def load_analysis(data, L, method, plot=True):
    daily_errors = []
    for _, day_df in data.groupby(data.date.dt.day):
        print(day_df.date)
        daily_errors.append(load_calculate_daily_error(L, day_df[L.column], method))

    daily_errors = np.asarray(daily_errors)

    error_df = pd.DataFrame()
    for step in range(daily_errors.shape[2]):
        step_err = daily_errors[:, :, step].flatten()
        error_df[step] = step_err

    error_df = error_df[
        (error_df < np.percentile(error_df, 95))
        & (error_df > np.percentile(error_df, 5))
    ]

    if plot:
        print("*" * 100)
        print(
            "Started load analysis with N={}, signal = {}, method = {}".format(
                L.N, L.column, method.__name__
            )
        )
        print("*" * 100 + "\n")
        print("****** Statistics ******")
        print(error_df.describe())
        print("\n" + "*" * 30 + " Lilliefors " + "*" * 30)
        print(
            "Lilliefors test-statistic:",
            lilliefors(daily_errors.flatten(), dist="norm")[1],
        )
        print_rmse(error_df)
        # plot_predictions(L, method)
        plot_boxplot(error_df, method.__name__)
        plot_daily_errors(daily_errors, method.__name__)
        plot_error_hist(daily_errors, method.__name__)
        plt.show()

    return error_df


def load_analysis_2(N, L, data, plot=True):
    error_df = pd.DataFrame()

    rmse = []
    days = 7
    time_skipped = 144 * days

    for i in range(time_skipped, data.shape[0] - N):
        current_time = data.date[i]
        L_true = data[
            (data["date"] > current_time)
            & (data["date"] <= (current_time + timedelta(minutes=10 * N)))
        ][L.column].values

        L_pred = L.get_previous_day(current_time, measurement=data.L.iloc[i], days=days)

        error_df[i - time_skipped] = L_true - L_pred
        rmse.append(np.sqrt(mean_squared_error(L_true, L_pred)))

    error_df = error_df.transpose()

    # error_df = error_df[
    #    (error_df < np.percentile(error_df, 90))
    # & (error_df > np.percentile(error_df, 10))
    # ]
    if plot:
        print("****** Load statistics ******")
        print(error_df.describe())
        print("\n" + "*" * 30 + " Lilliefors " + "*" * 30)
        print(
            "Lilliefors test-statistic:",
            lilliefors(error_df.values.flatten(), dist="norm")[1],
        )
        print_rmse(error_df)
        plot_boxplot(error_df, "Load")
        # plot_daily_errors(error_df.values, "PV")
        plot_error_hist(error_df.values, "Load")
        plt.show()
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
    for i in range(data.shape[0] - N):
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
            .iloc[:N]
        )
        # df = df.set_index("time")
        measurement = data[data.date == current_time]["PV"].iloc[0]
        df["PV_pred"] = pv.predict(
            df.airTemp.values, df.GHI_forecast.values, measurement
        )
        # df["PV_pred"] = pv.predict(df.temp.values, df.GHI_obs.values)
        # plt.plot(df.index, df["PV"], color="blue")
        # plt.plot(df.index, df["PV_pred"], color="red")
        error = df["PV"] - df["PV_pred"]
        rmse.append(np.sqrt(mean_squared_error(df["PV"], df["PV_pred"])))
        # error = np.divide(
        #    error, df["PV"], out=np.zeros_like(error), where=df["PV"] != 0
        # )
        error_df[i] = error

    error_df = error_df.transpose()
    # error_df = error_df[
    #    (error_df < np.percentile(error_df, 95))
    #    # & (error_df > np.percentile(error_df, 5))
    # ]

    if plot:
        print("****** PV statistics ******")
        print(error_df.describe())
        print("\n" + "*" * 30 + " Lilliefors " + "*" * 30)
        print(
            "Lilliefors test-statistic:",
            lilliefors(error_df.values.flatten(), dist="norm")[1],
        )
        print_rmse(error_df)
        plot_boxplot(error_df, "PV")
        # plot_daily_errors(error_df.values, "PV")
        plot_error_hist(error_df.values, "PV")
        plt.show()
    print(data.shape)
    print(error_df.shape)
    error_df["date"] = data.date
    error_df["rmse"] = rmse
    return error_df


def load_calculate_daily_error(L, day_load, method):
    """
    Calulates the errors for a given day
    """
    errors = []
    for step in range(day_load.shape[0] - L.N):
        gt = day_load.iloc[step : step + L.N + 1].values
        pred = method(gt[0], step)
        error = gt[1:] - pred  # / gt[0]
        errors.append(error)
    return np.asarray(errors)


def plot_predictions(L, method):
    """
    If groundtruth is provided to L, predictions can be plotted
    """
    plt.figure(figsize=(10, 5))
    for step in range(L.mean.shape[0] - L.N):
        plt.plot(
            range(step + 1, step + L.N + 1),
            method(L.true[0], step),
            color="red",
        )
    plt.plot(range(L.true.shape[0]), L.true, color="blue")
    plt.title("{} - Predictions vs groundtruth".format(method.__name__))
    plt.xlabel("Timestep")
    plt.ylabel("Power [kW]")


def print_rmse(error_df):
    """
    Prints the Root mean square error of a dataframe
    """
    print("RMSE total: ", np.sqrt(np.power(error_df.values, 2).mean()))


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
    L = Load(N, train_file, "L")
    test_data = pd.read_csv(train_file, parse_dates=["date"])
    load_errors = load_analysis_2(N, L, test_data, plot=True)

    observations = pd.read_csv(train_file, parse_dates=["date"]).fillna(0)
    solcast_forecasts = pd.read_csv(
        forecast_file, parse_dates=["time", "collected"]
    ).fillna(0)

    pv_errors = pv_analysis(
        N,
        PV,
        observations,
        solcast_forecasts,
        plot=False,
    )

    pv_errors = pv_errors.loc[~(pv_errors == 0).all(axis=1)]
    # load_errors = load_errors.loc[~(load_errors == 0).all(axis=1)]

    # pv_errors.to_csv("./data/pv_errors_date.csv")
    # load_errors.to_csv("./data/load_errors_date.csv")

    # return pv_errors, load_errors


if __name__ == "__main__":
    train_file = "./data/23.4_train.csv"
    test_file = "./data/23.4_test.csv"
    test_data_load = "./data/23.4_test.csv"
    estimate_errors(
        60,
        LinearPhotovoltaic(train_file),
        test_file,
        train_file,
        "./data/solcast_cleaned.csv",
    )
