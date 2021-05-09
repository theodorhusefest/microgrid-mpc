import numpy as np
from statsmodels.tools.eval_measures import rmse
from utils.helpers import parse_config


class SystemMetrics:
    def __init__(self, actions_per_hour=6):

        self.conf = parse_config()

        self.actions_per_hour = actions_per_hour
        self.battery_deg = self.conf["system"]["battery_cost"]
        self.peak_cost = self.conf["system"]["peak_cost"]
        self.C_max = self.conf["battery"]["C_MAX"]
        self.print_details = self.conf["print_details"]

        self.grid_cost = 0
        self.battery_cost = 0
        self.grid_max = 0
        self.primary_cost = 0
        self.pv_rmse = 0
        self.dependency_rate = -1
        self.consumption_rate = -1
        self.accumulated_error = 0

        self.computational_time = 0
        self.worst_case = 0
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

    def update_metrics(
        self, U, spot_price, e, Pgb_p, primary_buy, primary_sell, step_time
    ):
        """
        Updates all metrics
        """
        self.grid_cost += spot_price * (U[2] - 0.9 * U[3]) / self.actions_per_hour
        self.battery_cost += self.battery_deg * (U[0] + U[1]) / self.actions_per_hour
        self.grid_max = np.max([self.grid_max, Pgb_p])
        self.accumulated_error += np.abs(e)
        self.primary_cost += spot_price * (1.1 * primary_buy - 0.81 * primary_sell)
        self.steps += 1
        self.computational_time += step_time
        self.worst_case = np.max([self.worst_case, step_time])

    def print_metrics(self, SOC, E):
        """
        Prints the metrics
        """

        print("\nCosts")
        print("-" * 100)

        sec_cost = (
            self.grid_cost
            + self.battery_cost
            + self.peak_cost * self.grid_max
            + np.mean(E) * (SOC[0] - SOC[-1]) * self.C_max
        )

        print(
            "Seconday control cost = Grid cost + Battery cost + Peak cost + Battery offset"
        )
        print("Primary control cost = (Extra buy - Extra sell)  + 10% fee")
        print("-" * 100)
        print("Seconday control cost = {} NOK".format(np.around(sec_cost, 2)))
        print("Primary control cost = {} NOK".format(np.around(self.primary_cost, 2)))
        print("-" * 100)
        print("Total Costs = {}".format(np.around(self.primary_cost + sec_cost, 2)))

        print("\nComputational Time")
        print("-" * 100)
        print("Total time spent {}s".format(np.around(self.computational_time, 2)))
        print(
            "Average compuational time/step {}s".format(
                np.around(self.computational_time / self.steps, 4)
            )
        )
        print("Worst case computational time {}s".format(np.around(self.worst_case, 4)))
        print("-" * 100)

        if self.print_details:
            print("Final SOC: {}%".format(np.around(SOC, 2)))
            print("Battery Cost: {}".format(self.battery_cost))
            print("Max drawn from grid: {}".format(self.grid_max))
            print(
                "Self Dependency rate: {}%".format(
                    np.around(self.dependency_rate * 100, 1)
                )
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
