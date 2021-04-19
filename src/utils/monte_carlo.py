import time
from numba import njit
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import faiss
import matplotlib.pyplot as plt

from components.loads import Load
from components.PV import Photovoltaic


def monte_carlo_simulations(N, N_sim, forecast, errors):
    sims = np.zeros((N_sim, N))
    errors = np.c_[errors, np.zeros((errors.shape[0], N - errors.shape[1]))]

    if forecast.all() == 0:
        return sims

    for i in range(N_sim):
        ind = np.random.randint(0, errors.shape[0])

        sim = forecast + errors[ind, :N]  # (np.ones(N) - errors[ind, :N])

        sims[i] = sim

    return sims


def scenario_reduction(sims, N, Nr, branch_factor):
    """
    Uses Kmeans to cluster into N scenarios
    """
    N_scenarios = branch_factor ** Nr
    clusters = []
    if sims.all() == 0:
        return np.zeros((N_scenarios, N))
    for k in range(N):
        n_clusters = np.min([branch_factor ** (k + 1), N_scenarios])
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(sims[:, k].reshape(-1, 1))
        centers = kmeans.cluster_centers_

        if centers.shape[0] < N_scenarios:
            repeat_factor = int(N_scenarios / (branch_factor ** (k + 1)))
            centers = np.repeat(centers, repeat_factor)
        else:
            centers = centers.flatten()

        clusters.append(np.clip(np.sort(centers), 0, np.inf))

    return np.asarray(clusters).reshape(sims.shape[1], N_scenarios).T


def shuffle_dataframe(df):
    """
    Shufles values inside each column
    """
    df_shuff = pd.DataFrame()
    for k in range(len(df.columns)):
        df_shuff[k] = df[str(k)].sample(frac=1).values

    return df_shuff


def montecarlo_test():
    N = 18
    N_sim = 100
    N_scenarios = 4
    measurement = 40
    step = 40

    L = Load(N, "./data/8.4_train.csv", "L")
    load_errors = pd.read_csv("./data/load_errors.csv").drop(["Unnamed: 0"], axis=1)
    load_errors = shuffle_dataframe(load_errors)
    prediction = L.scaled_mean_pred(measurement, step)[1:]

    monte_carlo_jitted = njit()(monte_carlo_simulations)

    sims = monte_carlo_simulations(N, N_sim, prediction, load_errors.values)

    scenarios = scenario_reduction(sims, N, 2, 2)

    for scenario in scenarios:
        plt.plot(range(N), scenario)

    print(scenarios)
    plt.show()
