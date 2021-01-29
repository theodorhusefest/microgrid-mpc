import numpy as np
from statsmodels.tools.eval_measures import rmse


class SystemMetrics:
    def __init__(self, actions_per_hour=6):

        self.actions_per_hour = actions_per_hour
        self.battery_deg = 0.128

        self.grid_cost = 0
        self.battery_cost = 0
        self.wt_rmse = 0
        self.pv_rmse = 0

    def update_grid_cost(self, bought, sold, spot_price):
        """
        Bought from grid minus sold to grid
        """
        self.grid_cost += spot_price * (bought - sold) / self.actions_per_hour

    def update_battery_cost(self, charge, discharge, battery_cost):
        """
        Calculates battery degredation cost
        """

        self.battery_cost += (
            self.battery_deg * (charge + discharge) / self.actions_per_hour
        )

    def self_consumption_rate(self):
        """
        Amount of energy produced by the RES system to cover the load
        """
        pass

    def self_dependency_rate(self):
        """
        Relative amount of consumed power provided directly from PV-system
        or from PV after stored in battery
        """
        pass

    def update_metrics(self, U, spot_price):
        """
        Updates all metrics
        """

        self.grid_cost += spot_price * U[3] / self.actions_per_hour
        self.battery_cost += (
            self.battery_deg * (np.sum(np.abs([U[0:3]]))) / self.actions_per_hour
        )

    def print_metrics(self):
        """
        Prints the metrics
        """

        print("\nMETRICS")
        print("-" * 100)
        print("Grid Cost: {}".format(self.grid_cost))
        print("Battery Cost: {}".format(self.battery_cost))
        print("-" * 100)


def rmse_predictions(real, pred):
    return rmse(real[: len(pred)], pred)


def mean_absolute_error(real, pred):
    real, pred = np.asarray(real[: len(pred)]), np.asarray(pred)
    return np.mean(np.abs((real - pred) / real)) * 100
