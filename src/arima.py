"""Auto Regressive Integrated Moving Average"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")


class Arima:
    def __init__(self, column, order=(1, 1, 1)):
        self.df = (
            pd.read_csv("./data/arima-train.csv", parse_dates=["date"])
            .set_index("date")
            .filter(items=[column])
        )
        self.order = order
        self.column = column
        self.history = self.df[column].values
        self.start = self.df.index[0]
        self.end = self.df.index[-1]
        self.model = ARIMA(self.df, order=(1, 0, 0))
        self.model_fit = self.model.fit()

    def print_summary(self):
        print(self.model_fit.summary())

    def predict(self, hours):
        next_step = timedelta(minutes=10)
        return self.model_fit.predict(
            self.end + next_step, self.end + timedelta(hours=hours)
        ).values

    def update(self, observation):
        self.end = self.end + timedelta(minutes=10)
        new_datapoint = pd.DataFrame(
            index=[0],
            data={"date": self.end, self.column: observation},
            columns=["date", self.column],
        ).set_index("date")
        self.model_fit = self.model_fit.append(new_datapoint, refit=True)


if __name__ == "__main__":
    arima = Arima("PV")
    plt.plot(arima.df.index, arima.history)
    # arima.model_fit.plot_predict(arima.end, arima.end + timedelta(days=1))
    arima.update(0)
    print(arima.model_fit.predict(arima.end, arima.end + timedelta(hours=2)).values)
    # plt.show()
