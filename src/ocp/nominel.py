from casadi import *
from utils.helpers import parse_config


class NominelMPC:
    def __init__(self, T, N):

        self.T = T
        self.N = N
        conf = parse_config()
        conf_system = conf["system"]

        # Get system constants
        self.C_MAX = conf_system["C_MAX"]
        self.nb_c = conf_system["nb_c"]
        self.nb_d = conf_system["nb_d"]
        self.x_min = conf_system["x_min"]
        self.x_max = conf_system["x_max"]
        self.x_ref = conf_system["x_ref"]
        self.Pb_max = conf_system["Pb_max"]
        self.Pg_max = conf_system["Pg_max"]
        self.battery_cost = conf_system["battery_cost"]
        self.grid_cost = conf_system["grid_cost"]
        self.ref_cost = conf_system["ref_cost"]
        self.verbose = conf_system["verbose"]

        # Define symbolic variables
        self.x = SX.sym("x")
        self.u0 = SX.sym("u0")
        self.u1 = SX.sym("u1")
        self.u2 = SX.sym("u2")
        self.u3 = SX.sym("u3")

        self.u = vertcat(self.u0, self.u1, self.u2, self.u3)

        self.pv = SX.sym("pv", self.N)
        self.l1 = SX.sym("l1", self.N)
        self.l2 = SX.sym("l2", self.N)
        self.E = SX.sym("E", self.N)

        # Initialize system properties
        self.xdot = self.build_ode()
        self.L = None
        self.F = None
        self.solver = None

    def build_ode(self):
        """
        Returns the objective function.
        Can be dynamically updated
        """
        return (1 / self.C_MAX) * ((self.nb_c * self.u0) - (self.u1 / self.nb_d))

    def build_objective_function(self, e_spot):

        return (
            self.battery_cost * (self.u0 + self.u1)
            + e_spot * (self.u2 - self.u3)
            + self.grid_cost * (self.u2 + self.u3) ** 2
            + self.ref_cost * ((self.x_ref - self.x) * 100) ** 2
            + 100 * self.u0 * self.u1
            + 100 * self.u2 * self.u3
        )

    def build_integrator(self, e_spot):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function(
            "f", [self.x, self.u], [self.xdot, self.build_objective_function(e_spot)]
        )
        X0 = SX.sym("X0")
        U = SX.sym("U", 4)
        X = X0
        Q = 0
        for _ in range(M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT / 2 * k1, U)
            k3, k3_q = f(X + DT / 2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        self.F = Function("F", [X0, U], [X, Q], ["x0", "p"], ["xf", "qf"])

    def build_nlp(self):

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
        Xk = SX.sym("X0")
        w += [Xk]
        lbw += [0]
        ubw += [0]
        w0 += [0]

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = SX.sym("U_" + str(k), 4)
            w += [Uk]
            lbw += [0, 0, 0, 0]
            ubw += [self.Pb_max, self.Pb_max, self.Pg_max, self.Pg_max]
            w0 += [0, 0, 0, 0]
            F = self.build_integrator(self.E[k])
            Fk = self.F(x0=Xk, p=Uk)
            Xk_end = Fk["xf"]
            J = J + Fk["qf"]

            # New NLP variable for state at end of interval
            Xk = SX.sym("X_" + str(k + 1))
            w += [Xk]

            lbw += [self.x_min]
            ubw += [self.x_max]
            w0 += [0]

            # Add equality constraints
            g += [
                Xk_end - Xk,
                -Uk[0] + Uk[1] + Uk[2] - Uk[3] + self.pv[k] - self.l1[k] - self.l2[k],
            ]

            lbg += [0, 0]
            ubg += [0, 0]

        prob = {
            "f": J,
            "x": vertcat(*w),
            "g": vertcat(*g),
            "p": vertcat(self.pv, self.l1, self.l2, self.E),
        }
        if self.verbose:
            self.solver = nlpsol("solver", "ipopt", prob)
        else:
            opts = {
                "verbose_init": True,
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
        # J_opt = sol["f"].full().flatten()[0]

        x_opt = w_opt[0::5]
        u_opt = [w_opt[1::5], w_opt[2::5], w_opt[3::5], w_opt[4::5]]
        return x_opt, u_opt


if __name__ == "__main__":
    ocp = NominelMPC(1, 6)

    x, lbx, ubx, lbg, ubg = ocp.build_nlp()

    x[0] = 0.4
    lbx[0] = 0.4
    ubx[0] = 0.4

    xk_opt, Uk_opt, t_opt, J_opt = ocp.solve_nlp(
        [x, lbx, ubx, lbg, ubg],
        vertcat(
            np.asarray([50, 50, 50, 50, 50, 50]),
            np.asarray([0, 10, 50, 100, 200, 200]),
            np.asarray([10, 20, 100, 70, 50, 50]),
            np.asarray([1, 1.2, 1.3, 70, 50, 50]),
        ),
    )