import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MINUTES_PER_HOUR = 60
FONTSIZE = 15
FIGSIZE = (10, 5)
sn.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.8,
    rc={"lines.linewidth": 2},
    palette="tab10",
)


def get_formatter():
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = [
        "%y",  # ticks are mostly years
        "%b",  # ticks are mostly months
        "%d",  # ticks are mostly days
        "%H:%M",  # hrs
        "%H:%M",  # min
        "%S.%f",
    ]  # secs
    # these are mostly just the level above...
    formatter.zero_formats = [""] + formatter.formats[:-1]
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
    formatter.zero_formats[3] = "%d-%b"

    formatter.offset_formats = [
        "",
        "%Y",
        "%b %Y",
        "%d %b %Y",
        "%d %b %Y",
        "%d %b %Y %H:%M",
    ]
    return locator, formatter


def plot_from_df(
    df,
    columns,
    title="",
    ylabel="Power [kW]",
    upsample=False,
    legends=[],
    ax=None,
    xlabel="Date",
    logpath=None,
):
    df = df.copy()
    if upsample:
        df = df.resample("1T").ffill()

    if not ax:
        fig, ax = plt.subplots(figsize=FIGSIZE)

    for c in columns:
        ax.plot(df.index, df[c])

    locator, formatter = get_formatter()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legends)

    if logpath:
        plt.savefig(logpath + title + ".pdf", format="pdf")


def format_figure(
    fig,
    ax,
    time,
    title="",
    xlabel="Hours",
    ylabel="Power [kW]",
    legends=[],
    logpath=None,
):

    locator, formatter = get_formatter()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # ax.set_xticklabels(time)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if logpath:
        plt.savefig(logpath + title + ".pdf", format="pdf")
