import pandas as pd
import utils.plots as p
from simulations.spot_price import simulate_spotprice
from utils.helpers import parse_config


def get_data():
    """
    Returns all possible simulations
    """

    conf = parse_config()
    datafile = conf["datafile"]

    data = pd.read_csv(datafile)
    if "P1" in data.columns:
        PL = (data.P1 + data.P2).to_numpy()
    else:
        PL = data.PL.to_numpy()
    PV = data.PV.to_numpy()

    if "Spot_pris" in data.columns:
        grid_buy = grid_sell = data.Spot_pris.to_numpy()
    else:
        grid_buy = grid_sell = 1.5

    if conf["perfect_predictions"] or "PV_pred" not in data.columns:
        PV_pred = PV.copy()
        PL_pred = PL.copy()
    else:
        PV_pred = data.PV_pred.to_numpy()
        PL_pred = data.PL_pred.to_numpy()

    return PV, PV_pred, PL, PL_pred, grid_buy, grid_sell
