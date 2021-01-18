from casadi import *


class LinearOCP:
    def __init__(self, T, N):

        self.T = T
        self.N = N
        self.verbose = False

        self.C_MAX_0 = 500
        self.nb_0 = 0.8
        self.C_MAX_1 = 500
        self.nb_1 = 0.8
        self.x_min = 0.3
        self.x_max = 0.9
        self.x_ref = 0.7
        self.Pb_max = 1000
        self.Pg_max = 1000

        # Cost function variables
        self.c_b1 = 10
        self.c_b2 = 10
        self.grid_cost = 10
        self.e = 0.7

        # Define symbolic expressions
        self.x0 = SX.sym("x0")
        self.x1 = SX.sym("x1")
        self.x = vertcat(self.x0, self.x1)

        self.u0 = SX.sym("u0")
        self.u1 = SX.sym("u1")
        self.u2 = SX.sym("u2")
        self.u = vertcat(self.u0, self.u1, self.u2)

        # Loads
        self.l0 = SX.sym("l0", self.N)  # Essential
        self.l1 = SX.sym("l1", self.N)  # Non-essential

        # RES
        self.pv = SX.sym("pv", self.N)
        self.wt = SX.sym("wt", self.N)

        # Algebraic variables
        self.l = SX.sym("l")
        self.b = SX.sym("b")
        self.t = SX.sym("t")
        self.z = vertcat(self.l, self.b, self.t)

        # self.lv = SX.sym("lv")

        self.ode = self.build_ode()
        self.alg = self.build_algebraic_equations()
        self.L = self.build_objective_function()
        self.F = self.build_integrator()

    def build_ode(self):
        """
        Build the ODE used in the system
        """
        xdot_0 = (1 / self.C_MAX_0) * (self.nb_0 * self.u0)
        xdot_1 = (1 / self.C_MAX_1) * (self.nb_1 * self.u1)
        return vertcat(xdot_0, xdot_1)

    def build_objective_function(self):
        """
        Builds the objective function
        """
        return (
            self.c_b1 * self.u0 ** 2 + self.c_b2 * self.u0 ** 2 + self.e * self.u2 ** 2
        )

    def build_algebraic_equations(self):
        """
        Build the topology contraints as algebraic equations
        """
        fz_1 = self.wt + self.pv + self.u2 - self.l + self.b
        fz_2 = self.u0 + self.u1 - self.b
        fz_3 = self.l - self.l0 - self.l1
        return vertcat(fz_1, fz_2, fz_3)

    def build_integrator(self):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function("f", [self.x, self.u], [self.ode, self.L])
        X0 = SX.sym("X0", 2)
        U = SX.sym("U", 3)
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
            Uk = SX.sym("U_" + str(k), 3)
            w += [Uk]
            lbw += [0, 0, 0]
            ubw += [self.Pb_max, self.Pb_max, self.Pg_max]
            w0 += [0, 0, 0]

            Fk = self.F(x0=Xk, p=Uk)
            Xk_end = Fk["xf"]
            J = J + Fk["qf"]

            # New NLP variable for state at end of interval
            Xk = SX.sym("X_" + str(k + 1), 2)
            w += [Xk]

            lbw += [self.x_min, self.x_min]
            ubw += [self.x_max, self.x_max]
            w0 += [0, 0]

            # Add equality constraints
            g += [
                Xk_end - Xk,
                self.wt[k] + self.pv[k] + self.u2 - self.T,
                Uk[0] * Uk[1],
                Uk[2] * Uk[3],
            ]

            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

        prob = {
            "f": J,
            "x": vertcat(*w),
            "g": vertcat(*g),
            "p": vertcat(self.pv, self.l),
        }
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

    def test_integrator(self):
        """
        Tests a single point
        """
        Fk = self.F(x0=[0.4, 0.3], p=[100, 100, 0])
        print(Fk["xf"])


if __name__ == "__main__":
    ocp = LinearOCP(2, 12)
    ocp.test_integrator()