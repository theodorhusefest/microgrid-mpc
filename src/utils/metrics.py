import numpy as np
from statsmodels.tools.eval_measures import rmse


class Metrics:
    def __init__(self, actions_per_hour=6):

        self.actions_per_hour = actions_per_hour

        self.grid_cost = 0
        self.battery_cost = 0
        self.wt_rmse = 0
        self.pv_rmse = 0

    def update_grid_cost(self, U, spot_price):
        """
        Bought from grid minus sold to grid
        """
        bought = U[2] * spot_price
        sold = U[3] * spot_price

        self.grid_cost += (bought - sold) / self.actions_per_hour

    def update_battery_cost(self, U, battery_cost, actions_per_hour):
        """
        Calculates battery degredation cost
        """
        charge = U[0] * battery_cost
        discharge = U[1] * battery_cost

        self.battery_cost = (charge + discharge) / actions_per_hour


def rmse_predictions(real, pred):
    return rmse(real[: len(pred)], pred)


def mean_absolute_error(real, pred):
    real, pred = np.asarray(real[: len(pred)]), np.asarray(pred)
    return np.mean(np.abs((real - pred) / real)) * 100
