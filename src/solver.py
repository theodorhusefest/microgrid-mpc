from casadi import *
from utils.helpers import parse_config
from system import get_integrator


class OptiSolver:
    def __init__(self, T, N):

        self.T = T
        self.N = N

        conf = parse_config()

        # Get system constants
        self.C_MAX = conf["C_MAX"]
        self.nb_c = conf["nb_c"]
        self.nb_d = conf["nb_d"]
        self.x_min = conf["x_min"]
        self.x_max = conf["x_max"]
        self.x_ref = conf["x_ref"]
        self.Pb_max = conf["Pb_max"]
        self.Pg_max = conf["Pg_max"]
        self.battery_cost = conf["battery_cost"]
        self.grid_cost = conf["grid_cost"]
        self.grid_buy = conf["grid_buy"]
        self.grid_sell = conf["grid_sell"]
        self.ref_cost = conf["ref_cost"]
        self.verbose = conf["verbose"]

        # Define symbolic variables
        self.x = MX.sym("x")
        self.u0 = MX.sym("u0")
        self.u1 = MX.sym("u1")
        self.u2 = MX.sym("u2")
        self.u3 = MX.sym("u3")

        self.u = vertcat(self.u0, self.u1, self.u2, self.u3)

        # Initialize system properties
        self.xdot = self.build_ode()
        self.L = None
        self.integrator = None
        self.solver = None

    def build_ode(self):
        """
        Returns the objective function.
        Can be dynamically updated
        """
        return (1 / self.C_MAX) * ((self.nb_c * self.u0) - (self.u1 / self.nb_d))

    def build_objective_function(self):

        self.L = (
            self.battery_cost * (self.u0 + self.u1)
            + 10 * self.grid_buy * self.u2
            - 10 * self.grid_sell * self.u3
            + self.grid_cost * (self.u2 + self.u3) ** 2
            + self.ref_cost * ((self.x_ref - self.x) * 100) ** 2
        )

    def build_integrator(self):
        """
        Creates the given integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function("f", [self.x, self.u], [self.xdot, self.L])
        X0 = MX.sym("X0")
        U = MX.sym("U", 4)
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

    def solve_nlp(self, params):
        # Solve the NLP
        sol = self.solver(
            x0=params[0], lbx=params[0], ubx=params[0], lbg=params[0], ubg=params[0]
        )
        w_opt = sol["x"].full().flatten()

        x_opt = w_opt[0::5]
        u_opt = [w_opt[1::5], w_opt[2::5], w_opt[3::5], w_opt[4::5]]

        return x_opt, u_opt

    def build_nlp(self, x_inital, PV_pred, PL_pred):

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
        Xk = MX.sym("X0")
        w += [Xk]
        lbw += [x_inital]
        ubw += [x_inital]
        w0 += [x_inital]

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = MX.sym("U_" + str(k), 4)
            w += [Uk]
            lbw += [0, 0, 0, 0]
            ubw += [self.Pb_max, self.Pb_max, self.Pg_max, self.Pg_max]
            w0 += [0, 0, 0, 0]

            d0_k = PV_pred[k]
            d1_k = PL_pred[k]

            # Integrate till the end of the interval
            F = self.build_integrator()

            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk["xf"]
            J = J + Fk["qf"]

            # New NLP variable for state at end of interval
            Xk = MX.sym("X_" + str(k + 1))
            w += [Xk]
            lbw += [self.x_min]
            ubw += [self.x_max]
            w0 += []

            # Add equality constraints
            g += [
                Xk_end - Xk,
                -Uk[0] + Uk[1] + Uk[2] - Uk[3] + d0_k - d1_k,
                Uk[0] * Uk[1],
                Uk[2] * Uk[3],
            ]

            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

        prob = {"f": J, "x": vertcat(*w), "g": vertcat(*g)}
        if self.verbose:
            self.solver = nlpsol("solver", "ipopt", prob)
        else:
            opts = {
                "verbose_init": True,
                "ipopt": {"print_level": 0},
                "print_time": False,
            }
            self.solver = nlpsol("solver", "ipopt", prob, opts)

        return [w0, lbw, ubw, lbg, ubg]
