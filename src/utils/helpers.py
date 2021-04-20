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
    files = ["/config.yml", "/nominel_mpc.py", "/scenario_mpc.py"]
    # for f in files:
    #    shutil.copyfile(f, folderpath + f)
    return folderpath


def load_data():
    """"""

    conf = parse_config()
    datafile = conf["datafile"]

    data = pd.read_csv(datafile)
    if "L1" in data.columns:
        L1 = data.L1.to_numpy()
        L2 = data.L2.to_numpy()
    else:
        L = data.L.to_numpy()
    PV = data.PV.to_numpy()

    if "Spot_pris" in data.columns:
        grid_buy = data.Spot_pris.to_numpy()
    else:
        grid_buy = 1.5

    if "PV_pred" not in data.columns:
        PV_pred = PV.copy()
        L1_pred = L1.copy()
        L2_pred = L2.copy()
    else:
        PV_pred = data.PV_pred.to_numpy()
        L1_pred = data.L1_pred.to_numpy()
        L2_pred = data.L2_pred.to_numpy()

    return [PV, PV_pred, L1, L1_pred, L2, L2_pred, grid_buy]


def print_status(step, X, step_time, every=50):
    if step % every == 0:
        print("\nSTATUS - STEP {}".format(step))
        print("-" * 100)
        print("Current step took {}s".format(np.around(time.time() - step_time, 2)))
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


def check_constrain_satisfaction(u0, u1, u2, u3, pv, l):
    residual = -u0 + u1 + u2 - u3 + pv - l

    if residual > 1:
        print("Constraint breached")
        raise ValueError


def is_zero(x):
    return np.around(x, 3) == 0


def surplus_adjuster(e_hold, u):
    if e_hold >= u[2]:
        e_hold -= u[2]
        u[2] = 0
    else:
        u[2] -= e_hold
        e_hold = 0
        return u

    # Battery absorbs the rest of the surplus
    u[3] += e_hold
    return u


def deficit_adjuster(e_hold, u):
    e_hold = np.abs(e_hold)
    if e_hold >= u[3]:
        e_hold -= u[3]
        u[3] = 0
    else:
        u[3] -= e_hold
        e_hold = 0
        return u

    # Battery cover the rest with discharge
    u[2] += e_hold
    return u


def calculate_real_u(x, u, pv, l):
    """
    Calculates the real inputs based on topology contraint
    """
    e = -u[0] + u[1] + u[2] - u[3] + pv - l
    if np.around(e, 2) == 0:
        return 0, u
    if is_zero(e):
        pass
    elif e > 0:  # Suplus of energy
        u = surplus_adjuster(e, u)
    else:  # Need more energy
        u = deficit_adjuster(e, u)
    return e, u


def calculate_real_u_top(Uk, Tk, wt, pv, l1, l2):

    gen_error = np.around(wt + pv + Uk[6] - Uk[7] - Tk[0], 2)
    L_error = np.around(Tk[2] - l1 - l2, 2)

    if gen_error > 0:  # Surplus of energy in gen-node
        gen_error = np.abs(gen_error)
        if is_active(Uk[6]) and Uk[6] > gen_error:  # If buying energy buy less
            Uk[6] -= gen_error
        elif is_active(Uk[6]):  # Stop buying, then sell remaining
            Uk[7] += gen_error - Uk[6]
            Uk[6] = 0
        else:  # if selling -> sell more
            Uk[7] += gen_error

    elif gen_error < 0:  # Energy deficit in gen-node
        gen_error = np.abs(gen_error)
        if is_active(Uk[7]) and Uk[7] > gen_error:  # If selling -> sell less
            Uk[7] -= gen_error
        elif is_active(Uk[7]):
            Uk[6] += gen_error - Uk[7]
            Uk[7] = 0
        else:  # if buying -> buy more
            Uk[6] += gen_error

    if L_error > 0:  # Surplus of energy in load-node
        L_error = np.abs(L_error)
        Tk[2] -= L_error
        Tk[1] -= L_error

        if is_active(Uk[0]) and Uk[0] > L_error:
            Uk[0] -= L_error
        elif is_active(Uk[0]):
            Uk[1] += L_error - Uk[0]
            Uk[0] = 0
        else:
            Uk[1] += L_error

    elif L_error < 0:  # Energy deficit in load-node
        L_error = np.abs(L_error)
        Tk[2] += L_error
        Tk[1] += L_error

        if is_active(Uk[1]) and Uk[1] > L_error:
            Uk[1] -= L_error
        elif is_active(Uk[1]):
            Uk[0] += L_error - Uk[1]
            Uk[1] = 0
        else:
            Uk[0] += L_error

    assert np.around(wt + pv + Uk[6] - Uk[7] - Tk[0], 2) == 0
    assert np.around(Tk[0] + Tk[1] - Tk[2], 2) == 0
    assert np.around(Uk[0] - Uk[1] + Uk[2] - Uk[3] + Uk[4] - Uk[5] - Tk[1], 2) == 0
    assert np.around(Tk[2] - l1 - l2, 2) == 0

    return Uk, Tk


def is_active(x):
    return np.around(x, 3) != 0