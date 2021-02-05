import numpy as np
from casadi import *
import matplotlib.pyplot as plt


class LinearOCP:
    def __init__(self, T, N):
        self.T = T
        self.N = N
        self.verbose = False

        self.nx = 3
        self.nu = 8
        self.nt = 3
        self.nd = 4

        self.n = self.nx + self.nu + self.nt + self.nd

        self.C_MAX_0 = 300
        self.C_MAX_1 = 300
        self.C_MAX_2 = 300
        self.nb_0 = 0.9
        self.nb_1 = 0.7
        self.nb_2 = 0.6
        self.x_min = 0.3
        self.x_max = 0.9
        self.x_ref = 0.7
        self.Pb_max = 1000  # Max power from battery
        self.Pg_max = 1000  # Max power from grid
        self.P_max = 950  # Wire transmission maximum

        # Cost function variables
        self.c_b0 = 10
        self.c_b1 = 10
        self.c_b2 = 10
        self.grid_cost = 10
        self.e_spot = 0.7
        self.ref_cost = 1

        # Define symbolic expressions
        self.x0 = SX.sym("x0")
        self.x1 = SX.sym("x1")
        self.x2 = SX.sym("x2")
        self.x = vertcat(self.x0, self.x1, self.x2)

        self.PB1c = SX.sym("PB1c")  # PB1
        self.PB1d = SX.sym("PB1d")  # PB1
        self.PB2c = SX.sym("PB2c")  # PB2
        self.PB2d = SX.sym("PB2d")  # PB1
        self.PB3c = SX.sym("PB3c")  # PB3
        self.PB3d = SX.sym("PB3d")  # PB1
        self.PGb = SX.sym("PGb")  # PG
        self.PGs = SX.sym("PGs")  # PG
        self.u = vertcat(
            self.PB1c,
            self.PB1d,
            self.PB2c,
            self.PB2d,
            self.PB3c,
            self.PB3d,
            self.PGb,
            self.PGs,
        )

        # Disturbances
        # Loads
        self.l0 = SX.sym("l0", self.N)  # Essential
        self.l1 = SX.sym("l1", self.N)  # Non-essential

        self.E = SX.sym("E", self.N)

        # RES
        self.pv = SX.sym("pv", self.N)
        self.wt = SX.sym("wt", self.N)

        self.d = vertcat(self.l0, self.l1, self.pv, self.wt)

        self.ode = self.build_ode()
        self.L = None
        self.F = None

    def build_ode(self):
        """
        Build the ODE used in the system
        """
        xdot_0 = (1 / self.C_MAX_0) * (self.nb_0 * self.PB1c - self.PB1d / self.nb_0)
        xdot_1 = (1 / self.C_MAX_1) * (self.nb_1 * self.PB2c - self.PB2d / self.nb_1)
        xdot_2 = (1 / self.C_MAX_2) * (self.nb_2 * self.PB3c - self.PB3d / self.nb_2)
        return vertcat(xdot_0, xdot_1, xdot_2)

    def build_objective_function(self, e_spot):
        """
        Builds the objective function
        """
        return (
            self.c_b0 * (self.PB1c + self.PB1d)
            + self.c_b1 * (self.PB2c + self.PB2d)
            + self.c_b2 * (self.PB3c + self.PB3d)
            + self.ref_cost * ((self.x_ref - self.x0) * 100) ** 2
            + self.ref_cost * ((self.x_ref - self.x1) * 100) ** 2
            + self.ref_cost * ((self.x_ref - self.x2) * 100) ** 2
            + 10 * (self.PGb + self.PGs) ** 2
            + e_spot * (self.PGb - self.PGs)
            + 1000 * self.PB1c * self.PB1d
            + 1000 * self.PB2c * self.PB2d
            + 1000 * self.PB3c * self.PB3d
            + 1000 * self.PGb * self.PGs
        )

    def build_integrator(self, e_spot):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function(
            "f", [self.x, self.u], [self.ode, self.build_objective_function(e_spot)]
        )
        X0 = SX.sym("X0", 3)
        U = SX.sym("U", 8)
        X = X0
        Q = 0
        for _ in range(M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT / 2 * k1, U)
            k3, k3_q = f(X + DT / 2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        return Function("F", [X0, U], [X, Q], ["x0", "p"], ["xf", "qf"])

    def build_nlp(self):
        """
        Builds the multiple shooting NLP
        """

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = SX.sym("X0", 3)
        w += [Xk]
        lbw += [0] * self.nx
        ubw += [0] * self.nx
        w0 += [0] * self.nx

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = SX.sym("U_" + str(k), 8)
            w += [Uk]
            lbw += [0] * self.nu
            ubw += [self.Pb_max] * 6 + [self.Pg_max] * 2
            w0 += [0] * self.nu

            F = self.build_integrator(self.E[k])
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk["xf"]
            J = J + Fk["qf"]

            # New NLP variable for state at end of interval
            Xk = SX.sym("X_" + str(k + 1), 3)
            w += [Xk]

            lbw += [self.x_min, self.x_min, self.x_min]
            ubw += [self.x_max, self.x_max, self.x_max]
            w0 += [0] * self.nx

            # Topology contraints
            Tk = SX.sym("T_" + str(k), 3)
            w += [Tk]

            lbw += [-self.P_max, -self.P_max, -self.P_max]
            ubw += [self.P_max, self.P_max, self.P_max]
            w0 += [0] * self.nt

            # Add equality constraints
            # Tk[0] = P_T  Transformer
            # Tk[1] = P_B  Battery
            # Tk[2] = P_L  Loads
            eq_constraints = [
                Xk_end[0] - Xk[0],
                Xk_end[1] - Xk[1],
                Xk_end[2] - Xk[2],
                self.wt[k] + self.pv[k] + Uk[6] - Uk[7] - Tk[0],
                Tk[0] + Tk[1] - Tk[2],
                Uk[0] - Uk[1] + Uk[2] - Uk[3] + Uk[4] - Uk[5] - Tk[1],
                Tk[2] - self.l0[k] - self.l1[k],
            ]
            g += eq_constraints
            lbg += [0] * len(eq_constraints)
            ubg += [0] * len(eq_constraints)
        prob = {
            "f": J,
            "x": vertcat(*w),
            "g": vertcat(*g),
            "p": vertcat(self.wt, self.pv, self.l0, self.l1, self.E),
        }
        if self.verbose:
            self.solver = nlpsol("solver", "ipopt", prob)
        else:
            opts = {
                "verbose_init": False,
                "ipopt": {"print_level": 2},
                "print_time": False,
            }
            self.solver = nlpsol("solver", "ipopt", prob, opts)
        return [w0, lbw, ubw, lbg, ubg]

    def solve_nlp(self, params, p_ref):
        # Solve the NLP
        sol = self.solver(
            x0=params[0],
            lbx=params[1],
            ubx=params[2],
            lbg=params[3],
            ubg=params[4],
            p=p_ref,
        )
        w_opt = sol["x"].full().flatten()
        J_opt = sol["f"].full().flatten()[0]

        n = int(len(params[0]) / self.N)

        x_opt = [w_opt[self.nx + self.nu + i :: n] for i in range(self.nx)]
        u_opt = [w_opt[self.nx + i :: n] for i in range(self.nu)]
        t_opt = [w_opt[2 * self.nx + self.nu + i :: n] for i in range(self.nt)]

        self.test_mutual_exclusive(u_opt)
        return x_opt, u_opt, t_opt

    def test_mutual_exclusive(self, u_opt):

        for i in range(len(u_opt[0])):
            for j in range(0, len(u_opt), 2):
                if not (np.around(u_opt[j][i], 1) * np.around(u_opt[j + 1][i], 1) == 0):
                    print("U1 {}, U2 {}.".format(u_opt[j][i], 1, u_opt[j + 1][i], 1))


if __name__ == "__main__":

    print("Uncomment to run")
    """
    ocp = LinearOCP(1, 6)

    x, lbx, ubx, lbg, ubg = ocp.build_nlp()

    x[0] = 0.4
    x[1] = 0.7
    lbx[0] = 0.4
    lbx[1] = 0.7
    ubx[0] = 0.4
    ubx[1] = 0.7

    xk_opt, Uk_opt, t_opt, J_opt = ocp.solve_nlp(
        [x, lbx, ubx, lbg, ubg],
        vertcat(
            np.asarray([50, 50, 50, 50, 50, 50]),
            np.asarray([0, 10, 50, 100, 200, 200]),
            np.asarray([10, 20, 100, 70, 50, 50]),
            np.asarray([10, 10, 10, 10, 10, 10]),
        ),
    )

    p.plot_SOC(xk_opt[0], 1, title="Battery 1")
    p.plot_SOC(xk_opt[1], 1, title="Battery 2")

    p.plot_control_actions(Uk_opt, 1, 6, legends=["P_b1", "P_b2", "P_G"])

    p.plot_data(t_opt, title="Topology constrains", legends=["P_T", "P_B", "P_L"])

    plt.show()
    """
