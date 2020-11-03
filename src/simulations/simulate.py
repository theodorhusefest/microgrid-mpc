import utils.plots as p
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load
from simulations.spot_price import simulate_spotprice
from utils.helpers import parse_config, load_datafile


def get_simulations(logpath):
    """
    Returns all possible simulations
            conf[actions_per_hour],
        conf["simulation_horizon"],
        conf["simulations"],
        conf["datafile"],
        logpath,
        perfect_predictions=conf["perfect_predictions"],
        plot=conf["plot_predictions"],

    """

    conf = parse_config()
    datafile = conf["datafile"]
    days = int(conf["simulation_horizon"] / 24)

    PV_pred = simulate_pv_cell(
        samples_per_hour=conf["actions_per_hour"],
        sunrise=2,
        sunset=20,
        max_power=conf["simulations"]["pv_power"],
        days=days,
        plot=False,
        logpath=conf["logpath"],
        add_noise=conf["simulations"]["pv_noise"],
    )
    PL_pred = simulate_p_load(
        samples_per_hour=conf["actions_per_hour"],
        max_power=conf["simulations"]["pl_power"],
        days=days,
        plot=False,
        logpath=conf["logpath"],
        add_noise=conf["simulations"]["pl_noise"],
    )
    if datafile and conf["perfect_predictions"]:
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
            days, conf["actions_per_hour"], start_price=conf["simulations"]["grid_buy"]
        )
        grid_buy = simulate_spotprice(
            days, conf["actions_per_hour"], start_price=conf["simulations"]["grid_sell"]
        )

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
