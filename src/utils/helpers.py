import os
import shutil
import yaml
import time
import pandas as pd
import numpy as np
from datetime import datetime


def parse_config():
    """
    Parses and returns configfile as dict
    """
    with open("./config.yml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def create_logs_folder(rootdir="./logs/", foldername=""):
    """
    Creates a unique folder for the current run
    """
    now = datetime.now()
    time = now.strftime("%d.%m-%H:%M")
    folderpath = rootdir + time + "-" + foldername + "/"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    # Save files in logs
    files = ["/config.yml", "/main.py", "/scenario_mpc.py"]
    # for f in files:
    #    shutil.copyfile(f, folderpath + f)
    return folderpath


def print_status(step, X, step_time, every=50):
    if step % every == 0:
        print("\nSTATUS - STEP {}".format(step))
        print("-" * 100)
        print("Current step took {}s".format(np.around(step_time, 2)))
        for i, x in enumerate(X):
            print("x{}: {} %".format(i + 1, np.around(x, 2)))
        print("-" * 100)


def create_datafile(signals, names=[], logpath=None):
    """
    Saves all signals in a csvfile called signals.csv
    """
    data = {}
    for i in range(len(names)):
        data[names[i]] = signals[i]

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.transpose()
    return df


def is_zero(x):
    return np.around(x, 2) == 0


def get_scenario_from_file(scenarios, step, type_, filter_):
    return (
        scenarios[(scenarios.i == (step % 144)) & (scenarios.type == type_)]
        .filter(filter_)
        .values.flatten()
    )


def get_probability_from_file(scenarios, step, type_, filter_):
    return scenarios[(scenarios["i"] == (step % 144)) & (scenarios.type == type_)][
        "prob"
    ].values[0]


def surplus_adjuster_bat(e_hold, u):
    if e_hold >= u[1]:
        e_hold -= u[1]
        u[1] = 0
    else:
        u[1] -= e_hold
        e_hold = 0
        return u

    # Battery absorbs the rest of the surplus
    u[0] += e_hold
    return u


def deficit_adjuster_bat(e_hold, u):
    e_hold = np.abs(e_hold)
    if e_hold >= u[0]:
        e_hold -= u[0]
        u[0] = 0
    else:
        u[0] -= e_hold
        e_hold = 0
        return u

    # Battery cover the rest with discharge
    u[1] += e_hold
    return u


def surplus_adjuster_grid(e_hold, u):
    if e_hold >= u[2]:
        e_hold -= u[2]
        u[2] = 0
    else:
        u[2] -= e_hold
        e_hold = 0
        return u

    # Sell rest of surplus to grid
    u[3] += e_hold
    return u


def deficit_adjuster_grid(e_hold, u):
    e_hold = np.abs(e_hold)
    if e_hold >= u[3]:
        e_hold -= u[3]
        u[3] = 0
    else:
        u[3] -= e_hold
        e_hold = 0
        return u

    # Buy from grid to cover
    u[2] += e_hold
    return u


def primary_controller(x, uk, pv, l):
    """
    Calculates the real inputs based on topology contraint
    """

    u = np.copy(uk)
    e = -u[0] + u[1] + u[2] - u[3] + pv - l

    if is_zero(e):
        return 0, u
    elif e > 0:  # Suplus of energy
        u = surplus_adjuster_grid(e, u)
    else:  # Need more energy
        u = deficit_adjuster_grid(e, u)
    return e, u
