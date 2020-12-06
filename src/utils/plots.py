import numpy as np
import matplotlib.pyplot as plt

MINUTES_PER_HOUR = 60
FONTSIZE = 15
FIGSIZE = (10, 5)


def plot_SOC(
    SOC,
    timehorizon,
    logpath=None,
    title="State of charge",
    xlabel="Time [h]",
    ylabel="SOC [%]",
):
    """
    Plots the state of charge with hours on the time axis
    """
    plt.figure(figsize=FIGSIZE)
    plt.clf()
    plt.plot(np.linspace(0.0, timehorizon, SOC.shape[0]), SOC)
    plt.xlabel(xlabel, fontsize=FONTSIZE)

    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.title(title, fontsize=FONTSIZE)
    plt.ylim([0.25, 0.95])
    if logpath:
        plt.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")


def plot_control_actions(
    control,
    horizon,
    actions_per_hour,
    logpath=None,
    title="All controls",
    xlabel="Time [h]",
    ylabel="Power [kW]",
    legends=["Pb_charge", "Pb_discharge", "Pg_buy", "Pg_sell"],
):
    plt.figure(figsize=FIGSIZE)
    plt.clf()
    timehorizon = control[0].shape[0] * int(MINUTES_PER_HOUR / actions_per_hour)
    t = np.asarray(range(timehorizon))
    for u in control:
        u = np.asarray(u).flatten()
        u_plot = np.repeat(u, int(MINUTES_PER_HOUR / actions_per_hour))
        plt.plot(t, u_plot)

    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.xticks(
        np.linspace(0, timehorizon, horizon),
        range(horizon),
    )
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.title(title, fontsize=FONTSIZE)
    plt.legend(legends, fontsize=FONTSIZE)
    if logpath:
        plt.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")


def plot_data(
    series, logpath=None, title="", legends=[], xlabel="Time [h]", ylabel="Power [kW]"
):
    """
    Plots the given series
    """
    plt.figure(figsize=FIGSIZE)
    t = np.linspace(0, int(series[0].shape[0] / 6), series[0].shape[0])
    for serie in series:
        plt.plot(t, serie)
    plt.title(title, fontsize=FONTSIZE)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.legend(legends, fontsize=FONTSIZE)
    if logpath:
        plt.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")