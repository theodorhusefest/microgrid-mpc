import numpy as np
import pandas as pd

from casadi import SX, Function, vertcat


class Battery:
    def __init__(self, T, N, x_initial, nb_c, nb_d, C_MAX):

        self.xk_sim = x_initial
        self.xk_opt = x_initial
        self.nb_c = nb_c
        self.nb_d = nb_d
        self.C_MAX = C_MAX

        self.T = T
        self.N = N

        self.x = SX.sym("x")
        self.u = SX.sym("u", 2)

        self.ode = (1 / self.C_MAX) * (
            (self.nb_c * self.u[0]) - (self.u[1] / self.nb_d)
        )
        self.F = self.create_integrator()

        self.x_opt = [x_initial]
        self.x_sim = [x_initial]

    def create_integrator(self):

        M = 4
        DT = self.T / self.N / M
        f = Function("f", [self.x, self.u], [self.ode])
        X0 = SX.sym("X0")
        U = SX.sym("U", 2)
        X = X0
        for _ in range(M):
            k1 = f(X, U)
            k2 = f(X + DT / 2 * k1, U)
            k3 = f(X + DT / 2 * k2, U)
            k4 = f(X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Function("F", [X0, U], [X], ["x0", "p"], ["xf"])

    def simulate_SOC(self, x, uk):
        """
        Simulates SOC and saves the results
        """
        Fk = self.F(x0=self.xk_sim, p=uk)
        self.xk_sim = Fk["xf"].full().flatten()[0]

        if self.xk_sim < 0.22:
            self.xk_sim = 0.21
        elif self.xk_sim > 0.78:
            self.xk_sim = 0.79

        self.x_opt.append(x)
        self.xk_opt = x
        self.x_sim.append(self.xk_sim)

    def get_SOC(self):
        """
        Returns the last value of SOC
        """
        return np.around(self.xk_sim, 3)