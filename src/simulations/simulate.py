import pandas as pd
import utils.plots as p
from simulations.spot_price import simulate_spotprice
from utils.helpers import parse_config, load_datafile


def get_simulations(logpath):
    """
    Returns all possible simulations
    """

    conf = parse_config()
    datafile = conf["datafile"]

    data = pd.read_csv(datafile)
    PL = (data.P1 + data.P2).to_numpy()
    PV = data.PV.to_numpy()
    grid_buy = grid_sell = data.Spot_pris.to_numpy()

    if conf["perfect_predictions"] or "PV_pred" not in data.columns:
        PV_pred = PV
        PL_pred = PL
    else:
        PV_pred = data.PV_pred.to_numpy()
        PL_pred = data.PL_pred.to_numpy()

    if conf["plot_predictions"]:
        p.plot_data(
            [PV, PV_pred],
            logpath=logpath,
            title="Predicted vs real PV",
            legends=["PV", "Predicted PV"],
        )

        p.plot_data(
            [PL, PL_pred],
            logpath=logpath,
            title="Predicted vs real load",
            legends=["PL", "Predicted PL"],
        )

    return PV, PV_pred, PL, PL_pred, grid_buy, grid_sell
