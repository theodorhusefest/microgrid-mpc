import numpy as np
import matplotlib.pyplot as plt

MINUTES_PER_HOUR = 60


def plot_SOC(
    SOC,
    timehorizon,
    title="State of charge",
    xlabel="Time [h]",
    ylabel="SOC [%]",
):
    """
    Plots the state of charge with hours on the time axis
    """
    plt.figure()
    plt.clf()
    plt.plot(np.linspace(0.0, timehorizon, SOC.shape[0]), SOC)
    plt.xlabel(xlabel)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim([0.25, 0.95])


def plot_control_actions(
    control,
    timehorizon,
    actions_per_hour,
    title="All controls",
    xlabel="Time [h]",
    ylabel="Power [kW]",
    legends=["Pb_charge", "Pb_discharge", "Pg_buy", "Pg_sell"],
):
    plt.figure()
    plt.clf()
    t = np.asarray(range(timehorizon * MINUTES_PER_HOUR))
    for u in control:
        u = np.asarray(u).flatten()
        u_plot = np.repeat(u, int(MINUTES_PER_HOUR / actions_per_hour))
        plt.plot(t, u_plot)

    plt.xlabel(xlabel)
    plt.xticks(
        np.linspace(0, (timehorizon * MINUTES_PER_HOUR), timehorizon),
        range(timehorizon),
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legends)
