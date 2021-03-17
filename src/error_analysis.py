import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from components.loads import Load
from components.PV import Photovoltaic
from statsmodels.stats.diagnostic import lilliefors


def load_analysis(data, L, method, plot=True):
    daily_errors = []
    for day, day_df in data.groupby(data.date.dt.day):
        daily_errors.append(calculate_daily_error(L, day_df[L.column], method))

    daily_errors = np.asarray(daily_errors)

    error_df = pd.DataFrame()
    for step in range(daily_errors.shape[2]):
        step_err = daily_errors[:, :, step].flatten()
        error_df[step + 1] = step_err

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
        estimate_rmse(error_df)
        plot_predictions(L, method)
        plot_boxplot(error_df, method.__name__)
        plot_daily_errors(daily_errors, method.__name__)
        plot_error_hist(daily_errors, method.__name__)
        plt.show()

    return error_df


def pv_analysis(N, pv, data, forecasts, plot=True):
    """
    Calculates the error between prediction and observed production
    """
    daily_errors = []
    for day, day_df in data.groupby(data.date.dt.day):
        print(day)


def plot_predictions(L, method):
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


def estimate_rmse(error_df):
    def rmse(error):
        return np.sqrt(np.power(error, 2).mean())

    print("RMSE total: ", rmse(error_df.values))
    for i in range(error_df.shape[1]):
        print(
            "RMSE at prediction step {}: {}".format(
                i, np.around(rmse(error_df[i + 1].values), 2)
            )
        )


def calculate_daily_error(L, groundtruth, method):
    errors = []
    for step in range(groundtruth.shape[0] - L.N):
        gt = groundtruth.iloc[step : step + L.N + 1].values
        pred = method(gt[0], step)
        errors.append((pred - gt) / gt[1])
    return np.asarray(errors)


def plot_boxplot(df, name):
    ax = df.plot.box(figsize=(10, 5))
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Error")
    ax.set_title("{} - Error at Prediction Step".format(name))


def plot_error_hist(errors, name):
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=100)
    plt.title("{} - Error distribution".format(name))


def plot_daily_errors(daily_errors, name):
    daily_mean = daily_errors.mean(axis=0)
    plt.figure(figsize=(10, 5))
    for i in range(daily_mean.shape[0]):
        plt.scatter(range(i, i + daily_mean.shape[1]), daily_mean[i])
    plt.title("{} - Average prediction errors at timestep".format(name))
    plt.xlabel("Timestep")
    plt.ylabel("Error")


if __name__ == "__main__":
    N = 18
    # L = Load(N, "./data/loads_train.csv", "L2", groundtruth="./data/load_PV3.csv")
    # test_data = pd.read_csv("./data/data_oct20.csv", parse_dates=["date"]).iloc[::10]

    # load_analysis(test_data, L, L.scaled_mean_pred)
    # load_analysis(test_data, L, L.constant_pred)

    # Get data
    observations = pd.read_csv("./data/09.03_cleaned.csv", parse_dates=["date"])
    # observations = observations[observations["date"] >= datetime(2021, 3, 11)]
    solcast_forecasts = pd.read_csv(
        "./data/solcast_cleaned.csv", parse_dates=["time", "collected"]
    )

    pv_analysis(N, Photovoltaic(), observations, solcast_forecasts)
