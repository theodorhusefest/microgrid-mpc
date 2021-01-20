import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


class WindTurbine:
    def __init__(self):
        self.lower = 2
        self.upper = 16
        self.max_power = 810

        self.wind_ref = np.asarray(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
            ]
        )
        self.power_ref = np.asarray(
            [
                0.0,
                0.0,
                5.0,
                25.0,
                60.0,
                110.0,
                180.0,
                275.0,
                400.0,
                555.0,
                671.0,
                750.0,
                790.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
                810.0,
            ]
        )

        self.func = self.sigmoid
        self.params = self.curve_fitting(self.func)

    def get_power(self, wind):
        """
        Takes in either list or int
        Returns power
        """
        v = np.asarray(wind)

        return self.func(v, *self.params)

    def curve_fitting(self, func):
        """
        Uses Scipy to fit curve to wind-data
        """
        popt, _ = curve_fit(func, self.wind_ref, self.power_ref, method="dogbox")

        return popt

    def weibull_cdf(self, v, k, C):
        return self.max_power * (1 - np.exp(-k * (v / C)))

    def weibull_pdf(self, v, k, C):
        return (
            self.max_power
            * (k / C)
            * np.power((v / C), k - 1)
            * np.exp(np.power((-v / C), k))
        )

    def sigmoid(self, x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def four_params_logistic(self, v, a, b, c, d):
        return d + (a - d) / (a + np.power((v / c), b))

    def polynomial(self, v, a, b, c, d, e, f):
        return (
            f * np.power(v, 5)
            + e * np.power(v, 4)
            + a * np.power(v, 3)
            + b * np.power(v, 2)
            + c * v
            + d
        )


if __name__ == "__main__":
    wt = WindTurbine()

    wt.get_power(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
        ]
    )
