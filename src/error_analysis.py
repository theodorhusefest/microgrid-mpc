import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from components.loads import Load


def load_analysis(data, L, method, plot=True):

    # if method not in ["scaled_mean", "constant"]:
    #    raise ValueError("Prediction method does not exist.")

    daily_errors = []
    print(
        "Started load analysis with N={}, signal = {}, method = {}".format(
            L.N, L.column, method
        )
    )
    for day, day_df in data.groupby(data.date.dt.day):
        daily_errors.append(calculate_daily_error(L, day_df[L.column], method))

    daily_errors = np.asarray(daily_errors)

    error_df = pd.DataFrame()
    for step in range(daily_errors.shape[2]):
        step_err = daily_errors[:, :, step].flatten()
        error_df[step + 1] = step_err

    if plot:

        print("****** Statistics ******")
        print(error_df.describe())
        estimate_rmse(error_df)
        error_df.plot.box(figsize=(10, 5))
        plot_daily_errors(daily_errors)
        plot_error_hist(daily_errors)
        plt.show()

    return error_df


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
        errors.append((pred - gt[1:]))
    return np.asarray(errors)


def plot_error_hist(errors):
    plt.figure(figsize=(20, 10))
    plt.hist(errors.flatten(), bins=100)


def plot_daily_errors(daily_errors):
    daily_mean = daily_errors.mean(axis=0)
    plt.figure(figsize=(10, 5))
    for i in range(daily_mean.shape[0]):
        plt.scatter(range(i, i + daily_mean.shape[1]), daily_mean[i])
    plt.title("Average prediction errors at timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Error")


if __name__ == "__main__":
    N = 12
    L = Load(N, "./data/loads_train.csv", "L1")
    test_data = pd.read_csv("./data/data_oct20.csv", parse_dates=["date"]).iloc[::10]

    load_analysis(test_data, L, L.get_scaled_mean)
    load_analysis(test_data, L, L.get_constant_pred)
