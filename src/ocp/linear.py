from casadi import *
import matplotlib.pyplot as plt


class LinearOCP:
    def __init__(self, T, N):

        self.T = T
        self.N = N
        self.verbose = False

        self.nx = 3
        self.nu = 4
        self.nt = 3
        self.nd = 4

        self.n = self.nx + self.nu + self.nt + self.nd

        self.C_MAX_0 = 300
        self.C_MAX_1 = 300
        self.C_MAX_2 = 300
        self.nb_0 = 0.5
        self.nb_1 = 0.8
        self.nb_2 = 0.6
        self.x_min = 0.3
        self.x_max = 0.9
        self.x_ref = 0.7
        self.Pb_max = 1000  # Max power from battery
        self.Pg_max = 1000  # Max power from grid
        self.P_max = 1000  # Wire transmission maximum

        # Cost function variables
        self.c_b0 = 10
        self.c_b1 = 10
        self.c_b2 = 10
        self.grid_cost = 10
        self.e_spot = 0.7

        # Define symbolic expressions
        self.x0 = SX.sym("x0")
        self.x1 = SX.sym("x1")
        self.x2 = SX.sym("x2")
        self.x = vertcat(self.x0, self.x1, self.x2)

        self.u0 = SX.sym("u0")  # PB1
        self.u1 = SX.sym("u1")  # PB2
        self.u2 = SX.sym("u2")  # PB3
        self.u3 = SX.sym("u3")  # PG
        self.u = vertcat(self.u0, self.u1, self.u2, self.u3)

        # Disturbances
        # Loads
        self.l0 = SX.sym("l0", self.N)  # Essential
        self.l1 = SX.sym("l1", self.N)  # Non-essential

        # RES
        self.pv = SX.sym("pv", self.N)
        self.wt = SX.sym("wt", self.N)

        self.d = vertcat(self.l0, self.l1, self.pv, self.wt)

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
        xdot_0 = -(1 / self.C_MAX_0) * (self.nb_0 * self.u0)
        xdot_1 = -(1 / self.C_MAX_1) * (self.nb_1 * self.u1)
        xdot_2 = -(1 / self.C_MAX_2) * (self.nb_2 * self.u2)
        return vertcat(xdot_0, xdot_1, xdot_2)

    def build_objective_function(self):
        """
        Builds the objective function
        """
        return (
            self.c_b0 * self.u0 ** 2
            + self.c_b1 * self.u1 ** 2
            + self.c_b2 * self.u2 ** 2
            + 100 * self.e_spot * self.u3 ** 2
        )

    def build_algebraic_equations(self):
        """
        Build the topology contraints as algebraic equations
        """
        fz_1 = self.wt + self.pv + self.u2 - self.l + self.b
        fz_2 = self.u0 + self.u1 + self.u3 - self.b
        fz_3 = self.l - self.l0 - self.l1
        return vertcat(fz_1, fz_2, fz_3)

    def build_integrator(self):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function("f", [self.x, self.u], [self.ode, self.L])
        X0 = SX.sym("X0", 3)
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
        Xk = SX.sym("X0", 3)
        w += [Xk]
        lbw += [0] * self.nx
        ubw += [0] * self.nx
        w0 += [0] * self.nx

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = SX.sym("U_" + str(k), 4)
            w += [Uk]
            lbw += [-self.Pb_max, -self.Pb_max, -self.Pb_max, -self.Pg_max]
            ubw += [self.Pb_max, self.Pb_max, self.Pb_max, self.Pg_max]
            w0 += [0] * self.nu

            Fk = self.F(x0=Xk, p=Uk)
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
            g += [
                Xk_end[0] - Xk[0],
                Xk_end[1] - Xk[1],
                self.wt[k] + self.pv[k] + Uk[3] - Tk[0],
                Tk[0] + Tk[1] - Tk[2],
                Uk[0] + Uk[1] + Uk[2] - Tk[1],
                Tk[2] - self.l0[k] - self.l1[k],
            ]
            lbg += [0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0]
        prob = {
            "f": J,
            "x": vertcat(*w),
            "g": vertcat(*g),
            "p": vertcat(self.wt, self.pv, self.l0, self.l1),
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
        print(w)

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

        x_opt = [w_opt[7::n], w_opt[8::n], w_opt[9::n]]
        u_opt = [w_opt[3::n], w_opt[4::n], w_opt[5::n], w_opt[6::n]]
        t_opt = [w_opt[10::n], w_opt[11::n], w_opt[12::n]]
        return x_opt, u_opt, t_opt, J_opt

    def test_integrator(self):
        """
        Tests a single point
        """
        print("Testing integrator")
        Fk = self.F(
            x0=[0.4, 0.4],
            p=[
                -100,
                0,
                0,
            ],
        )
        print(Fk["xf"])


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
