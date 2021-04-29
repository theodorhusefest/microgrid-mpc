"""Auto Regressive Integrated Moving Average"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from datetime import timedelta
import pmdarima as pm


import warnings

warnings.filterwarnings("ignore")


class Arima:
    def __init__(
        self,
        column,
        order=(1, 1, 2),
        seasonal_order=(1, 1, 1, 144),
    ):
        self.df = (
            pd.read_csv("./data/23.4_train.csv", parse_dates=["date"])
            .set_index("date")
            .filter(items=[column])
            # .diff()
            # .diff(144)
            .dropna()
        )

        if column == "PV":
            self.df.iloc[-10:] = 1.2
        self.order = order
        self.seasonal_order = seasonal_order
        self.column = column
        self.x = self.df[column].values
        self.clean_data()
        self.df[column] = self.x
        self.end = self.df.index[-1]
        self.model = ARIMA(
            self.df,
            order=self.order,
            # seasonal_order=self.seasonal_order,
            # enforce_stationarity=True,
            dates=self.df.index,
        )

        self.model_fit = self.model.fit()

    def clean_data(self):
        Nc = 50
        self.x = np.convolve(self.x, np.ones(Nc) / Nc, mode="same")

    def predict(self, hours):
        pred = self.model_fit.predict(
            self.end,
            self.end + timedelta(hours=hours) - timedelta(minutes=10),
        )
        return np.clip(pred.values, 0, np.inf)

    def update(self, observation):
        self.end = self.end + timedelta(minutes=10)
        new_datapoint = pd.DataFrame(
            index=[0],
            data={"date": self.end, self.column: observation},
            columns=["date", self.column],
        ).set_index("date")
        self.model_fit = self.model_fit.append(new_datapoint, refit=True)

    def print_summary(self):
        print(self.model_fit.summary())

    def check_stationarity(self):
        result = adfuller(self.x)
        print("ADF Statistic: %f" % result[0])
        print("p-value: %f" % result[1])
        print("Critical Values:")
        for key, value in result[4].items():
            print("\t%s: %.3f" % (key, value))

    def plot_autocorrelation(self):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.x, lags=150, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.x, lags=150, ax=ax2)
        plt.show()

    def grid_search(self):

        best = np.inf
        for p in range(1, 7):
            for q in range(1, 7):
                for P in range(1, 2):
                    for Q in range(1, 2):
                        model = ARIMA(
                            self.x,
                            order=(p, 1, q),
                            # seasonal_order=(P, 1, Q, 144),
                            dates=self.df.index,
                        )
                        model_fit = model.fit()
                        print(
                            "({},1,{}), ({},1,{},144) - AIC = {}.".format(
                                p, q, P, Q, model_fit.aic
                            )
                        )
                        aic = model_fit.aic
                        if aic < best:
                            best = aic
                            best_model = (p, q)
        print(best, best_model)


if __name__ == "__main__":
    # auto_model = AutoArima("PV")
    # auto_model.decompose()

    arima = Arima("L", order=(2, 1, 4), seasonal_order=(1, 1, 1, 144))
    arima.print_summary()
    # arima.grid_search()
    test_data = pd.read_csv("./data/23.4_test.csv", parse_dates=["date"])
    N = 12
    for i in range(20):
        prediction = arima.predict(int(N / 6))
        arima.update(test_data.iloc[i]["L"])
        plt.plot(range(i, i + N), prediction, color="red")
        plt.plot(range(i, i + N), test_data.iloc[i : i + N]["L"].values, color="blue")
        print(i)
    plt.show()

    # arima.check_stationarity()
    # arima.plot_autocorrelation()
    # plt.plot(arima.df.index, arima.history)
    # arima.model_fit.plot_predict(arima.end, arima.end + timedelta(days=1))
    # plt.show()
