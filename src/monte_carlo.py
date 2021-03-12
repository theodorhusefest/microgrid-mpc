import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from error_analysis import load_analysis
from components.loads import Load


def monte_carlo_simulations(measurement, step, N, N_sim, component, method, test_data):
    start = time.time()
    error_df = load_analysis(test_data, component, method, plot=False)
    sims = []
    forecast = method(measurement, step)
    step = time.time()
    # print("Load analysis took {}s.\n".format(np.around(step - start, 2)))
    errors = shuffle_dataframe(error_df)

    for i in range(N_sim):
        ind = np.random.randint(0, errors.shape[0])

        sim = measurement * (np.ones(N) - errors.iloc[ind])

        sims.append(np.asarray(sim))
        # plt.plot(range(N), sim)

    # print("Monte Carlo took {}s.".format(np.around(time.time() - step, 2)))
    # print("Total time {}".format(np.around(time.time() - start, 2)))
    # plt.show()
    return np.asarray(sims)


def scenario_reduction(sims, N_scenarios):
    """
    Uses Kmeans to cluster into N scenarios
    """
    kmeans = KMeans(n_clusters=N_scenarios)
    clusters = []
    for k in range(sims.shape[1]):
        kmeans.fit(sims[:, k].reshape(-1, 1))
        clusters.append(kmeans.cluster_centers_)

    return np.asarray(clusters).reshape(sims.shape[1], N_scenarios).T


def get_monte_carlo_scenarios(
    measurement, step, N, N_sim, N_scenarios, component, method, test_data, plot=False
):

    sims = monte_carlo_simulations(
        measurement, step, N, N_sim, component, method, test_data
    )
    scenarios = scenario_reduction(sims, N_scenarios)
    if plot:
        for k in range(N_scenarios):
            plt.plot(range(N), scenarios[k], marker="o")
        plt.show()
    return scenarios


def shuffle_dataframe(df):
    """
    Shufles values inside each column
    """
    df_shuff = pd.DataFrame()
    for k in range(len(df.columns)):
        df_shuff[k + 1] = df.iloc[k + 1].sample(frac=1).values

    return df_shuff


if __name__ == "__main__":
    N = 36
    N_sim = 100
    N_scenarios = 10

    test_data = pd.read_csv("./data/data_oct20.csv", parse_dates=["date"]).iloc[::10]
    L = Load(N, "./data/loads_train.csv", "L")

    sims = monte_carlo_simulations(10, 50, N, N_sim, L, L.scaled_mean_pred, test_data)

    scenarios = scenario_reduction(sims, N_scenarios)

    plt.figure()

    plt.show()
