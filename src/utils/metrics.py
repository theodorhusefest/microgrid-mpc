import numpy as np
from statsmodels.tools.eval_measures import rmse


class SystemMetrics:
    def __init__(self, actions_per_hour=6):

        self.actions_per_hour = actions_per_hour
        self.battery_deg = 0.128

        self.grid_cost = 0
        self.battery_cost = 0
        self.grid_max = 0
        self.pv_rmse = 0
        self.dependency_rate = -1
        self.consumption_rate = -1
        self.accumulated_error = 0

        self.steps = 0

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

    def calculate_consumption_rate(self, Pgs, pv):
        """
        Amount of energy produced by the RES system to cover the load
        """
        self.consumption_rate = 1 - (np.sum(Pgs) / np.sum(pv[0 : len(Pgs)]))

    def calculate_dependency_rate(self, Pgb, l):
        """
        Relative amount of consumed power provided directly from PV-system
        or from PV after stored in battery
        """
        self.dependency_rate = 1 - (np.sum(Pgb) / np.sum(l[0 : len(Pgb)]))

    def update_metrics(self, U, spot_price, e):
        """
        Updates all metrics
        """

        self.grid_cost += spot_price * (U[2] - U[3]) / self.actions_per_hour
        self.battery_cost += (
            self.battery_deg * (np.sum(np.abs([U[0:1]]))) / self.actions_per_hour
        )
        self.grid_max = np.max([self.grid_max, U[2]])
        self.accumulated_error += np.abs(e)

    def print_metrics(self, SOC):
        """
        Prints the metrics
        """

        print("\nMETRICS")
        print("-" * 100)
        print("Final SOC: {}%".format(np.around(SOC, 2)))
        print("Grid Cost: {}".format(self.grid_cost))
        print("Battery Cost: {}".format(self.battery_cost))
        print("Max drawn from grid: {}".format(self.grid_max))
        print(
            "Self Dependency rate: {}%".format(np.around(self.dependency_rate * 100, 1))
        )
        print(
            "Self Consumption rate: {}%".format(
                np.around(self.consumption_rate * 100, 1)
            )
        )
        print("Accumulated Error: ", self.accumulated_error)
        print("-" * 100)


def rmse_predictions(real, pred):
    return rmse(real[: len(pred)], pred)


def mean_absolute_error(real, pred):
    real, pred = np.asarray(real[: len(pred)]), np.asarray(pred)
    return np.mean(np.abs((real - pred) / real)) * 100
