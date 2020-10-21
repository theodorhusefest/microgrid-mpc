import utils.plots as p
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load
from simulations.spot_price import simulate_spotprice
from utils.helpers import load_datafile


def get_simulations(
    actions_per_hour,
    days,
    conf,
    datafile,
    logpath,
    perfect_predictions=False,
    plot=False,
):
    """
    Returns all possible simulations

    """
    PV_pred = simulate_pv_cell(
        samples_per_hour=actions_per_hour,
        sunrise=2,
        sunset=20,
        max_power=conf["pv_power"],
        days=days,
        plot=False,
        logpath=logpath,
        add_noise=conf["pv_noise"],
    )
    PL_pred = simulate_p_load(
        samples_per_hour=actions_per_hour,
        max_power=conf["pl_power"],
        days=days,
        plot=False,
        logpath=logpath,
        add_noise=conf["pl_noise"],
    )
    if datafile and perfect_predictions:
        PV, PL, spotprice = load_datafile(datafile)
        grid_buy = spotprice
        grid_sell = spotprice
        PV_pred = PV
        PL_pred = PL

    elif datafile:
        PV, PL, spotprice = load_datafile(datafile)
        grid_buy = spotprice
        grid_sell = spotprice

    else:
        grid_buy = simulate_spotprice(
            days, actions_per_hour, start_price=conf["grid_buy"]
        )
        grid_buy = simulate_spotprice(
            days, actions_per_hour, start_price=conf["grid_sell"]
        )

    if plot:
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
