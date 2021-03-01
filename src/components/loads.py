import numpy as np
import pandas as pd


class Load:
    def __init__(self, N, train_file, column, actions_per_hour=6, groundtruth=None):
        self.N = N
        self.groundtruth = groundtruth
        self.train_file = train_file
        self.resolution = 60 / actions_per_hour
        self.column = column

        self.df = pd.read_csv(self.train_file, parse_dates=["date"])
        self.mean = self.get_mean_day()

        if isinstance(groundtruth, pd.Series):
            self.true = self.groundtruth.values
        elif isinstance(groundtruth, pd.DataFrame):
            self.true = self.groundtruth[column].values
        elif isinstance(groundtruth, str):
            self.true = pd.read_csv(self.groundtruth)[self.column].values
        elif isinstance(groundtruth, np.ndarray):
            self.true = self.groundtruth

    def get_mean_day(self):
        """
        Extracts the days and returns the average day
        """

        num_datapoints = 24 * 60 / self.resolution
        days = []
        grouped = self.df.groupby([self.df.date.dt.floor("d")], as_index=False)
        for _, group in grouped:
            if len(group) != num_datapoints:
                continue
            days.append(group[self.column].values)

        return np.asarray(days).mean(axis=0)

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

    def get_measurement(self, step):
        if self.groundtruth is None:
            raise ValueError("Groundtruth not provided")

        return self.true[step]


if __name__ == "__main__":
    l1 = Load(6, "./data/loads_train.csv", "L1", groundtruth="./data/load_PV3.csv")
    print(l1.get_groundtruth(50))