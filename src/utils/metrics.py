import logging
import numpy as np
from datetime import datetime
from statsmodels.tools.eval_measures import rmse
from utils.helpers import parse_config


class SystemMetrics:
    def __init__(self, actions_per_hour=6):

        self.logger = logging.getLogger("Metrics")
        f_handler = logging.FileHandler("./logs/metrics.log")
        s_handler = logging.StreamHandler()
        f_handler.setLevel(logging.DEBUG)
        s_handler.setLevel(logging.INFO)

        s_format = logging.Formatter("%(message)s")
        f_format = logging.Formatter("%(message)s")
        s_handler.setFormatter(s_format)
        f_handler.setFormatter(f_format)

        self.logger.addHandler(s_handler)
        self.logger.addHandler(f_handler)

        self.conf = parse_config()
        self.actions_per_hour = actions_per_hour
        self.battery_deg = self.conf["system"]["battery_cost"]
        self.peak_cost = self.conf["system"]["peak_cost"]
        self.old_grid_fee = self.conf["system"]["old_grid_fee"]

        assert (
            self.peak_cost * self.old_grid_fee == 0
        ), "Either peak or old cost has to be zero"

        self.C_max = self.conf["battery"]["C_MAX"]
        self.print_details = self.conf["print_details"]

        self.logger.info("\n \n")
        self.logger.info("Run started at {}".format(datetime.now()))
        self.logger.info(
            "Sim-time: {}, Pred-horizon: {}, Branches: {}, Perfect Predictions: {}".format(
                str(self.conf["simulation_horizon"]),
                str(self.conf["prediction_horizon"]),
                str(self.conf["branch_factor"]),
                str(self.conf["perfect_predictions"]),
            )
        )
        self.logger.info("-" * 100)

        self.grid_cost = 0
        self.battery_cost = 0
        self.grid_max = 0
        self.old_system_cost = 0
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
        self.old_system_cost += 0.06 * U[2]
        self.steps += 1
        self.computational_time += step_time
        self.worst_case = np.max([self.worst_case, step_time])

    def print_metrics(self, SOC, E):
        """
        Prints the metrics
        """

        self.logger.info("\nCosts")
        self.logger.info("-" * 100)

        sec_cost = (
            self.grid_cost
            + self.battery_cost
            + self.peak_cost * self.grid_max
            + self.old_system_cost
            + np.mean(E) * (SOC[0] - SOC[-1]) * self.C_max
        )

        self.logger.info(
            "Seconday control cost = Grid cost + Battery cost + Peak cost + Battery offset"
        )
        self.logger.info("Primary control cost = (Extra buy - Extra sell)  + 10% fee")
        self.logger.info("-" * 100)
        self.logger.info(
            "Seconday control cost = {} NOK".format(np.around(sec_cost, 2))
        )
        self.logger.info(
            "Primary control cost = {} NOK".format(np.around(self.primary_cost, 2))
        )
        self.logger.info("-" * 100)
        self.logger.info(
            "Total Costs = {}".format(np.around(self.primary_cost + sec_cost, 2))
        )
        self.logger.info("\nComputational Time")
        self.logger.info("-" * 100)
        self.logger.info(
            "Total time spent {}s".format(np.around(self.computational_time, 2))
        )
        self.logger.info(
            "Average compuational time/step {}s".format(
                np.around(self.computational_time / self.steps, 4)
            )
        )
        self.logger.info(
            "Worst case computational time {}s".format(np.around(self.worst_case, 4))
        )
        self.logger.info("-" * 100)

        # Extra info beeing logged
        self.logger.info("Final SOC: {}%".format(np.around(SOC[-1], 2)))
        self.logger.info("Grid Cost: {}".format(self.grid_cost))
        self.logger.info("Old System Cost: {}".format(self.old_system_cost))
        self.logger.info("Battery Cost: {}".format(self.battery_cost))
        self.logger.info("Max drawn from grid: {}".format(self.grid_max))
        self.logger.info(
            "Self Dependency rate: {}%".format(np.around(self.dependency_rate * 100, 1))
        )
        self.logger.info(
            "Self Consumption rate: {}%".format(
                np.around(self.consumption_rate * 100, 1)
            )
        )
        self.logger.info("Accumulated Error: {}".format(self.accumulated_error))
        self.logger.info("-" * 100)


def rmse_predictions(real, pred):
    return rmse(real[: len(pred)], pred)


def mean_absolute_error(real, pred):
    real, pred = np.asarray(real[: len(pred)]), np.asarray(pred)
    return np.mean(np.abs((real - pred) / real)) * 100
