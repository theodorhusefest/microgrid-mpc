"""Auto Regressive Integrated Moving Average"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import warnings

warnings.filterwarnings("ignore")


class Arima:
    def __init__(self, column, datapoint):
        df = (
            pd.read_csv("./data/oct20_10_red.csv")
            .set_index("date")
            .filter(items=[column])
        )
        self.history = df[column].values
        self.history = np.append(self.history, datapoint)

        self.model = ARIMA(self.history, order=(1, 0, 1))
        self.model_fit = self.model.fit(disp=0)
        self.predictions = [datapoint]

    def print_summary(self):
        print(self.model_fit.summary())

    def predict(self):
        pred = self.model_fit.forecast()[0][0]
        self.predictions.append(pred)
        return pred

    def update(self, observation):
        self.history = np.append(self.history, observation)
        self.model = ARIMA(self.history, order=(5, 1, 0))
        self.model_fit = self.model.fit(disp=0)
