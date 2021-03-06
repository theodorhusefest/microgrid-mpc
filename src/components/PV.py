from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d


class Photovoltaic:

    # Parameters from Vinod2018
    def __init__(
        self,
        Pm=800,
        Im=8.36,
        Vm=37.38,
        Voc=90,
        Isc=9.06,
        Ns=72,
        A=1.3,
        Ki=0.058,
        Eg=1.1,
    ):
        self.V_value = 0
        self.I_value = 0
        self.Pm = Pm  # Power max at STC
        self.Im = Im  # Current max at STC
        self.Vm = Vm  # Voltage max at STC
        self.Voc = Voc  # Open circuit voltage at STC
        self.Isc = Isc  # Short circuit current
        self.Ns = Ns  # Number of cells connected in series
        self.A = A  # Diode ideality factor
        self.Ki = Ki  # Model constant
        self.Eg = Eg  # Energy band gap
        self.rs = 0  # Equivalent series resistance
        self.rsh = 0  # Shunt resistance
        self.K = 1.3805e-23  # Boltzmann
        self.q = 1.602e-19  # Electron charge
        self.Tref = 298.15  # Ref temp STC of 25C in Kelvin
        self.Gref = 1000  # Ref solar STC

        self.solver = None

        self.estimate()
        self.create_nlp()

    def estimate(self):
        I = SX.sym("I")
        V = SX.sym("V")
        Rs = SX.sym("Rs")
        Rsh = SX.sym("Rsh")

        Irs = SX.sym("Irs")
        Is = SX.sym("Is")
        Iph = SX.sym("Iph")

        Irs = self.Isc / (
            np.exp((self.q * self.Voc) / (self.Ns * self.K * self.A * self.Tref)) - 1
        )
        Is = Irs
        Iph = self.Isc

        g = I - (
            Iph
            - Is
            * (
                np.exp(self.q * (V + I * Rs) / (self.Ns * self.K * self.A * self.Tref))
                - 1
            )
            - (V + I * Rs) / Rsh
        )
        f = power(self.Pm - I * V, 2)

        nlp_prob = {"f": f, "x": vertcat(I, V, Rs, Rsh), "g": g}

        s_opts = {"ipopt": {"print_level": 0}, "print_time": False}

        solver = nlpsol("solver", "ipopt", nlp_prob, s_opts)

        x0 = DM.ones(1, 4)
        lbx = 1e-3 * DM.ones(1, 4)
        ubx = DM([1e3, 1e3, 100, 100])
        lbg = 0
        ubg = 0

        res_nlp = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        self.I_value = res_nlp["x"][0]
        self.V_value = res_nlp["x"][1]
        self.rs = res_nlp["x"][2]
        self.rsh = res_nlp["x"][3]

    def create_nlp(self):
        I = SX.sym("I")
        V = SX.sym("V")
        T = SX.sym("T")
        G = SX.sym("G")

        Irs = SX.sym("Irs")  # Reverse saturation current
        Is = SX.sym("Is")  # Saturation current
        Iph = SX.sym("Iph")  # Photovoltaic current

        Irs = self.Isc / (
            np.exp((self.q * self.Voc) / (self.Ns * self.K * self.A * T)) - 1
        )
        Is = (
            Irs
            * power(T / self.Tref, 3)
            * np.exp((self.q * self.Eg) / (self.A * self.K) * (1 / self.Tref - 1 / T))
        )
        Iph = G / self.Gref * (self.Isc + self.Ki * (T - self.Tref))

        g = I - (
            Iph
            - Is
            * (np.exp(self.q * (V + I * self.rs) / (self.Ns * self.K * self.A * T)) - 1)
            # - (V + I * self.rs) / self.rsh
        )
        f = -I * V

        nlp_prob = {"f": f, "x": vertcat(I, V), "g": g, "p": vertcat(T, G)}

        s_opts = {"ipopt": {"print_level": 0}, "print_time": False}

        self.solver = nlpsol("solver", "ipopt", nlp_prob, s_opts)

    def solve_prob(self, T, G):
        x0 = DM([0, 0])
        lbx = DM([0, 0])
        ubx = DM([1e3, 1e3])
        lbg = 0
        ubg = 0
        param = vertcat(273.15 + T, G)

        res_nlp = self.solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=param)
        self.I_value = res_nlp["x"][0]
        self.V_value = res_nlp["x"][1]

        return (self.V_value * self.I_value).full().flatten()[0]

    def predict(self, T, G):

        assert len(T) == len(G)
        return np.asarray([self.solve_prob(T[i], G[i]) for i in range(len(T))])


class LinearPhotovoltaic:
    def __init__(self, train_file):
        self.train_file = train_file

        self.train = (
            pd.read_csv(train_file, parse_dates=["date"]).set_index("date").fillna(0)
        )

        X = self.train.filter(["airTemp", "GHI"])
        y = self.train.filter(["PV"])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )

        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)

    def get_mean_day(self, column):
        """
        Extracts the days and returns the average day
        """

        num_datapoints = 24 * 60 / 10
        days = []
        grouped = self.train.groupby([self.train.index.floor("d")], as_index=False)
        for _, group in grouped:
            if len(group) != num_datapoints:
                continue
            days.append(group[column].values)

        return np.asarray(days).mean(axis=0)

    def predict(self, temp, GHI, measurement=None):
        """
        Main function used to predict the PV
        """
        pred = self.model.predict(np.c_[temp, GHI]).flatten()
        pred = np.clip(pred, 0, np.inf)
        if np.max(pred) > 1 and measurement:
            time_steps = 16
            pred_weight = np.append(
                np.linspace(0, 1, time_steps), np.ones(len(pred) - time_steps)
            )
            measurement_weight = np.append(
                np.linspace(1, 0, time_steps), np.zeros(len(pred) - time_steps)
            )
            pred = measurement_weight * measurement + pred_weight * pred

        return pred
