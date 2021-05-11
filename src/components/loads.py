import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class Load:
    def __init__(
        self, N, train_file, column, current_time, actions_per_hour=6, groundtruth=None
    ):
        self.N = N
        self.groundtruth = groundtruth
        self.train_file = train_file
        self.resolution = 60 / actions_per_hour
        self.column = column

        self.df = pd.read_csv(self.train_file, parse_dates=["date"]).fillna(0)
        self.update_statistics(current_time)

        if isinstance(groundtruth, pd.Series):
            self.true = self.groundtruth.values
        elif isinstance(groundtruth, pd.DataFrame):
            self.true = self.groundtruth[column].values
        elif isinstance(groundtruth, str):
            self.true = pd.read_csv(self.groundtruth)[self.column].values
        elif isinstance(groundtruth, np.ndarray):
            self.true = self.groundtruth

    def update_statistics(self, current_time):
        """
        Extracts the days and returns the average day
        """

        num_datapoints = 24 * 60 / self.resolution
        week = []
        weekend = []
        pred_start = current_time - timedelta(days=15)
        one_week_df = self.df[(self.df["date"] > pred_start)]
        grouped = one_week_df.groupby([one_week_df.date.dt.floor("d")], as_index=False)
        for day, group in grouped:
            if len(group) != num_datapoints:
                continue

            if is_weekday(day):
                week.append(group[self.column].values)
            else:
                weekend.append(group[self.column].values)

        self.week_upper, self.week_lower = get_minmax(week)
        self.weekend_upper, self.weekend_lower = get_minmax(weekend)
        self.week_mean = np.tile(np.mean(week, axis=0), 2)
        self.weekend_mean = np.tile(np.mean(weekend, axis=0), 2)

    def get_prediction_mean(self, step):
        """
        Returns the mean for the next N steps
        """
        return self.mean[step : step + self.N]

    def scaled_mean_pred(self, measurement, step):
        """
        Returns the mean for the next N steps, scaled to current measurement
        """
        pred = (measurement / self.mean[step]) * self.mean[step : step + self.N + 1][1:]
        return pred

    def constant_pred(self, measurement, step):
        """
        Returns the current measurement as prediction for N
        """
        return measurement * np.ones(self.N)

    def perfect_pred(self, step):
        """
        Returns the groundtruth for k+1 to N
        """
        return self.true[step : step + self.N + 1][1:]

    def get_statistic_scenarios(self, current_time, step):
        """
        Extracts the 90 percentiles scenarios based on two previous weeks
        """
        # Update stats at midnight
        if current_time.hour == 0 and current_time.minute == 0:
            self.update_statistics(current_time)

        if is_weekday(current_time):
            return (
                self.week_mean[step % 144 + 1 : step % 144 + 1 + self.N],
                self.week_lower[step % 144 + 1 : step % 144 + 1 + self.N],
                self.week_upper[step % 144 + 1 : step % 144 + 1 + self.N],
            )
        else:
            return (
                self.weekend_mean[step % 144 + 1 : step % 144 + 1 + self.N],
                self.weekend_lower[step % 144 + 1 : step % 144 + 1 + self.N],
                self.weekend_upper[step % 144 + 1 : step % 144 + 1 + self.N],
            )

    def get_minmax_day(self, current_time, step):
        """
        Extracts the days and returns the average day
        """

        num_datapoints = 24 * 60 / self.resolution
        days = []
        pred_start = current_time - timedelta(days=8)
        one_week_df = self.df[(self.df["date"] > pred_start)]
        grouped = one_week_df.groupby([one_week_df.date.dt.floor("d")], as_index=False)
        for _, group in grouped:
            if len(group) != num_datapoints:
                continue
            days.append(group[self.column].values)
        min_ = np.asarray(days).min(axis=0)
        max_ = np.asarray(days).max(axis=0)
        return (
            np.tile(min_, 2)[step % 144 + 1 : step % 144 + 1 + self.N],
            np.tile(max_, 2)[step % 144 + 1 : step % 144 + 1 + self.N],
        )

    def get_previous_day(self, current_time, days=1, measurement=None, rounds=1):
        """
        Returns the same values the day before
        """

        preds = []
        for i in range(rounds):
            pred_start = current_time - timedelta(days=(i + 1) * days)
            pred = self.df[
                (self.df["date"] > pred_start)
                & (self.df["date"] <= (pred_start + timedelta(minutes=10 * self.N)))
            ][self.column].values

            preds.append(pred)

        pred = np.mean(preds, axis=0)
        if np.min(pred) > 1 and measurement:
            pred = self.interpolate_prediction(pred, measurement)

        return pred

    def interpolate_prediction(self, pred, measurement):
        """
        Returns a linear combination of prediction and measurement
        """

        pred_weight = np.append(np.linspace(0, 1, 12 + 1), np.ones(len(pred) - 12))
        measurement_weight = np.append(
            np.linspace(1, 0, 12 + 1), np.zeros(len(pred) - 12)
        )
        pred = np.append(np.asarray([measurement]), pred)

        pred = (measurement_weight * measurement + pred_weight * pred)[1:]
        return pred


def is_weekday(day):
    if day.weekday() < 5:
        return True
    return False


def get_percentiles(array, pu=99, pl=1):

    upper = np.percentile(array, pu, axis=0)
    lower = np.percentile(array, pl, axis=0)

    return np.tile(lower, 2), np.tile(upper, 2)


def get_minmax(array):

    upper = np.max(array, axis=0)
    lower = np.min(array, axis=0)

    return np.tile(lower, 2), np.tile(upper, 2)


if __name__ == "__main__":
    l1 = Load(6, "./data/loads_train.csv", "L1", groundtruth="./data/load_PV3.csv")
    print(l1.get_groundtruth(50))