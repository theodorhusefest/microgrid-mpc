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

        if groundtruth:
            self.true = pd.read_csv(self.groundtruth)[self.column].values

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

    def get_scaled_mean(self, measurement, step):
        """
        Returns the mean for the next N steps, scaled to current measurement
        """
        return (measurement / self.mean[step]) * self.mean[step : step + self.N]

    def get_groundtruth(self, step):
        if self.groundtruth == None:
            print("Groundtruth not provided")
            raise ValueError

        return self.true[step : step + self.N]


if __name__ == "__main__":
    l1 = Load(6, "./data/loads_train.csv", "L1", groundtruth="./data/load_PV3.csv")
    print(l1.get_groundtruth(50))