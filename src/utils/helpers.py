import os
import shutil
import yaml
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


def create_folder(folderpath):
    """
    Creates a folder
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


def create_logs_folder(rootdir="./logs/", foldername=""):
    """
    Creates a unique folder for the current run
    """
    now = datetime.now()
    time = now.strftime("%d.%m-%H:%M")
    folderpath = rootdir + time + "-" + foldername + "/"
    create_folder(folderpath)

    # Save files in logs
    files = ["./config.yml", "./main.py", "./solver.py"]
    for f in files:
        shutil.copyfile(f, folderpath + f)
    return folderpath


def load_data():
    """"""

    conf = parse_config()
    datafile = conf["datafile"]

    data = pd.read_csv(datafile)
    if "P1" in data.columns:
        L1 = data.P1.to_numpy()
        L2 = data.P2.to_numpy()
    else:
        L = data.PL.to_numpy()
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


def print_stats(PV, L, PV_pred, PL_pred):
    print(
        "Predicted energy produced {}, predicted energy consumed {}".format(
            np.sum(PV_pred), np.sum(PL_pred)
        )
    )

    print(
        "Actual energy produced {}, actual energy consumed {}".format(
            np.sum(PV), np.sum(L)
        )
    )
    print("Predicted energy surplus/deficit:", np.sum(PV_pred) - np.sum(PL_pred))
    print("Actual energy surplus/deficit:", np.sum(PV) - np.sum(L))


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


if __name__ == "__main__":
    parse_config()