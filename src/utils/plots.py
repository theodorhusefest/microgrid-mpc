import numpy as np
import matplotlib.pyplot as plt

MINUTES_PER_HOUR = 60
FONTSIZE = 15
FIGSIZE = (10, 5)


def plot_SOC(
    SOC,
    timehorizon,
    logpath=None,
    title="State of Charge",
    xlabel="Time [h]",
    ylabel="SOC [%]",
    ax=None,
):
    """
    Plots the state of charge with hours on the time axis
    """
    if not ax:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(np.linspace(0.0, timehorizon, SOC.shape[0]), SOC * 100)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)

    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_ylim([25, 95])
    if logpath:
        try:
            fig.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")
        except:
            pass


def plot_control_actions(
    control,
    horizon,
    actions_per_hour,
    logpath=None,
    title="All Controls",
    xlabel="Time [h]",
    ylabel="Power [kW]",
    legends=["Pb_charge", "Pb_discharge", "Pg_buy", "Pg_sell"],
    ax=None,
):
    if not ax:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    timehorizon = control[0].shape[0] * int(MINUTES_PER_HOUR / actions_per_hour)
    t = np.asarray(range(timehorizon))
    for u in control:
        u = np.asarray(u).flatten()
        u_plot = np.repeat(u, int(MINUTES_PER_HOUR / actions_per_hour))
        ax.plot(t, u_plot)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_xticks(np.linspace(0, timehorizon, horizon))
    ax.set_xticklabels(range(horizon))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.legend(legends, fontsize=FONTSIZE)
    if logpath:
        try:
            fig.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")
        except:
            pass


def plot_data(
    series,
    logpath=None,
    title="",
    legends=[],
    xlabel="Time [h]",
    ylabel="Power [kW]",
    ax=None,
):
    """
    Plots the given series
    """
    if not ax:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    t = np.linspace(0, int(series[0].shape[0] / 6), series[0].shape[0])
    for serie in series:
        ax.plot(t, serie)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(legends, fontsize=FONTSIZE)
    if logpath:
        try:
            fig.savefig("{}-{}".format(logpath, title + ".eps"), format="eps")
        except:
            pass


def plot_predictions_subplots(PV, pv_preds, PL, pl_preds, logpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    plot_data(
        [PV[: len(pv_preds)], pv_preds],
        legends=["Measured PV", "Predicted PV"],
        title="Predicted vs Measured PV Production",
        xlabel="",
        ax=ax1,
    )
    plot_data(
        [PL[: len(pl_preds)], pl_preds],
        legends=["Measured Load", "Predicted Load"],
        title="Predicted vs Measured Load Demand",
        ax=ax2,
    )
    if logpath:
        fig.savefig("{}{}".format(logpath, "PV_PL_2_4" + ".eps"), format="eps")


def plot_SOC_control_subplots(x, u, horizon, logpath=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    plot_SOC(x, horizon, ax=ax1)
    plot_control_actions(u, horizon, 6, ax=ax2)

    if logpath:
        fig.savefig("{}{}".format(logpath, "SOC_U_2_4" + ".eps"), format="eps")
