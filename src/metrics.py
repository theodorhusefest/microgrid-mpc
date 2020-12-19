import numpy as np
from statsmodels.tools.eval_measures import rmse


def net_spending_grid(U, spot_price, actions_per_hour):
    "Calculates how much buy, sell to the grid."
    bought = U[2] * spot_price
    sold = U[3] * spot_price

    return (bought - sold) / actions_per_hour


def net_change_battery(u0, u1):
    change_c = 0
    change_d = 0

    for i in range(len(u0) - 1):
        change_c += np.abs(u0[i + 1] - u0[i])
        change_d += np.abs(u1[i + 1] - u1[i])

    return (change_c + change_d) / u0.shape[0]


def net_cost_battery(U, battery_cost, actions_per_hour):
    charge = U[0] * battery_cost
    discharge = U[1] * battery_cost

    return (charge + discharge) / actions_per_hour


def rmse_predictions(real, pred):
    return rmse(real[: len(pred)], pred)


def mean_absolute_error(real, pred):
    real, pred = np.asarray(real[: len(pred)]), np.asarray(pred)
    return np.mean(np.abs((real - pred) / real)) * 100
