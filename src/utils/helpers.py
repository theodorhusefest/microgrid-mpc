import os
import shutil
import yaml
from datetime import datetime


def parse_config():
    """
    Parses and returns configfile
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


def create_logs_folder(rootdir="./logs/"):
    """
    Creates a unique folder for the current run
    """
    now = datetime.now()
    time = now.strftime("%d.%m-%H:%M/")
    folderpath = rootdir + time
    create_folder(folderpath)

    # Save files in logs
    files = ["./config.yml", "./main.py", "./open_loop.py"]
    for f in files:
        shutil.copyfile(f, folderpath + f)
    return folderpath


if __name__ == "__main__":
    parse_config()