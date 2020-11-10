import os
import shutil
import yaml
import pandas as pd
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


def load_datafile(datapath):
    """
    Reads a CSV file with real data, splits and returns it
    """
    data = pd.read_csv(datapath)
    PL = (data.P1 + data.P2).to_numpy()
    try:
        return [
            data.PV.to_numpy(),
            PL,
            data.Spot_pris.to_numpy(),
            data.PV_pred.to_numpy(),
            data.PL_pred.to_numpy(),
        ]
    except AttributeError:
        return [data.PV.to_numpy(), PL, data.Spot_pris.to_numpy()]


def save_datafile(signals, names=[], logpath=None):
    """
    Saves all signals in a csvfile called signals.csv
    """
    if not logpath:
        return

    data = {}
    for i in range(len(names)):
        data[names[i]] = signals[i]

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.transpose()
    df.to_csv(logpath + "signals.csv")


if __name__ == "__main__":
    parse_config()