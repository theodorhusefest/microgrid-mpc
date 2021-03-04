from casadi import *
from casadi.tools import *
from utils.helpers import parse_config
from pprint import pprint


class ScenarioMPC:
    def __init__(self, T, N, N_scenarios):

        self.T = T
        self.N = N
        self.N_scenarios = N_scenarios
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

        self.states = struct_symSX([entry("SOC")])
        self.inputs = struct_symSX(
            [
                entry("Pbc"),
                entry("Pbd"),
                entry("Pgb"),
                entry("Pgs"),
            ]
        )

        self.w = struct_symSX(
            [
                entry("states", struct=self.states, repeat=self.N),
                entry("inputs", struct=self.inputs, repeat=self.N - 1),
            ]
        )
        self.data = struct_symSX([entry("pv"), entry("l1"), entry("l2"), entry("E")])
        self.all_data = struct_symSX([entry("data", struct=self.data, repeat=self.N)])

        self.scenario = struct_symSX(
            [entry("w", struct=self.w), entry("data", struct=self.data, repeat=self.N)]
        )
        scenarios = []

        for k in range(self.N_scenarios):
            scenarios.append(entry("scenario" + str(k), struct=self.scenario))

        self.scenarios = struct_symSX(scenarios)

        self.E = SX.sym("E", self.N)

        # Initialize system properties
        self.ode = self.build_ode()
        self.L = None
        self.F = None
        self.solver = None

        # Keep optimal solutions

        self.SOC = np.asarray([])
        self.Pbc = np.asarray([])
        self.Pbd = np.asarray([])
        self.Pgs = np.asarray([])
        self.Pgb = np.asarray([])

    def get_SOC_opt(self):
        """
        Returns the last updated SOC
        """
        return self.SOC[-1]

    def get_u_opt(self):
        """
        Returns the last calculated optimal U
        """
        return np.asarray([self.Pbc[-1], self.Pbd[-1], self.Pgb[-1], self.Pgs[-1]])

    def build_ode(self):
        """
        Returns the objective function.
        """
        return (1 / self.C_MAX) * (
            (self.nb_c * self.inputs["Pbc"]) - self.inputs["Pbd"] / self.nb_d
        )

    def update_forecasts(self, pv, l1, l2, E):
        """
        Creates datastruct with relevant data
        """

        data_struct = self.all_data(0)
        for k in range(self.N):
            data_struct["data", k, "pv"] = pv[k]
            data_struct["data", k, "l1"] = l1[k]
            data_struct["data", k, "l2"] = l2[k]
            data_struct["data", k, "E"] = E[k]

        return data_struct

    def build_objective_function(self, e_spot):

        return (
            self.battery_cost * (self.inputs["Pbc"] + self.inputs["Pbd"])
            + e_spot * (self.inputs["Pgb"] - self.inputs["Pgs"])
            + self.grid_cost * (self.inputs["Pgb"] + self.inputs["Pgs"]) ** 2
            + 100 * self.inputs["Pbc"] * self.inputs["Pbd"]
            + 100 * self.inputs["Pgb"] * self.inputs["Pgs"]
        )

    def build_integrator(self, e_spot):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function(
            "f",
            [self.states, self.inputs],
            [self.ode, self.build_objective_function(e_spot)],
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
        return Function("F", [X0, U], [X, Q], ["x0", "p"], ["xf", "qf"])

    def solve_scenario_tree(self, x, forecasts):

        w0, lbw, ubw, ubg, lbg = self.build_single_scenario()
        SOC = []
        Pbc = []
        Pbd = []
        Pgs = []
        Pgb = []
        for k in range(self.N_scenarios):
            data = self.update_forecasts(
                forecasts["scenario" + str(k), "data", :, "pv"],
                forecasts["scenario" + str(k), "data", :, "l1"],
                forecasts["scenario" + str(k), "data", :, "l2"],
                forecasts["scenario" + str(k), "data", :, "E"],
            )

            sol = self.solver(
                x0=w0,
                lbx=lbw,
                ubx=ubw,
                lbg=lbg,
                ubg=ubg,
                p=data,
            )
            w_opt = sol["x"].full().flatten()

            j_opt = sol["g"].full().flatten()
            w_opt = self.w(w_opt)
            SOC.append(w_opt["states", :, "SOC"])
            Pbc.append(w_opt["inputs", :, "Pbc"])
            Pbd.append(w_opt["inputs", :, "Pbd"])
            Pgb.append(w_opt["inputs", :, "Pgb"])
            Pgs.append(w_opt["inputs", :, "Pgs"])

        return np.asarray(SOC)[:, 1].mean(), [
            np.asarray(Pbc)[:, 0].mean(),
            np.asarray(Pbd)[:, 0].mean(),
            np.asarray(Pgb)[:, 0].mean(),
            np.asarray(Pgs)[:, 0].mean(),
        ]

    def build_single_scenario(self):

        J = 0

        lbw = self.w(0)
        ubw = self.w(0)
        lbw["states", :, "SOC"] = self.x_min
        ubw["states", :, "SOC"] = self.x_max
        ubw["inputs", :, "Pbc"] = self.Pb_max
        ubw["inputs", :, "Pbd"] = self.Pb_max
        ubw["inputs", :, "Pgs"] = self.Pg_max
        ubw["inputs", :, "Pgb"] = self.Pg_max

        w0 = self.w(0)
        g = []
        lbg = []
        ubg = []

        F = self.build_integrator(0.50)

        for k in range(self.N - 1):

            states_k = self.w["states", k]
            inputs_k = self.w["inputs", k]
            data_k = self.all_data["data", k]

            # System dynamics
            Fk = F(x0=states_k, p=inputs_k)
            Xk_end = Fk["xf"]
            J += Fk["qf"]
            # Stage costs
            # J += self.battery_cost * (
            #    self.w["inputs", k, "Pbc"] * self.w["inputs", k, "Pbd"]
            # )
            # J += self.all_data["data", k, "E"] * (self.w["inputs", k, "Pgb"])
            # J += (
            #    self.grid_cost
            #    * (self.w["inputs", k, "Pgb"] + self.w["inputs", k, "Pgs"]) ** 2
            # )

            # Equality Contraints
            g += [Xk_end - self.w["states", k + 1]]
            g += [
                -self.w["inputs", k, "Pbc"]
                + self.w["inputs", k, "Pbd"]
                + self.w["inputs", k, "Pgb"]
                - self.w["inputs", k, "Pgs"]
                + self.all_data["data", k, "pv"]
                - self.all_data["data", k, "l1"]
                - self.all_data["data", k, "l2"]
            ]
            lbg += [0] * (Xk_end.size(1) + 1)
            ubg += [0] * (Xk_end.size(1) + 1)

        prob = {"f": J, "x": self.w, "g": vertcat(*g), "p": self.all_data}
        if self.verbose:
            self.solver = nlpsol("solver", "ipopt", prob)
        else:
            opts = {
                "verbose_init": True,
                "ipopt": {"print_level": 2},
                "print_time": False,
            }
            self.solver = nlpsol("solver", "ipopt", prob, opts)

        return [w0, lbw, ubw, ubg, lbg]

    def solve_nlp(self, params, data):
        # Solve the NLP
        sol = self.solver(
            x0=params[0],
            lbx=params[1],
            ubx=params[2],
            lbg=params[3],
            ubg=params[4],
            p=data,
        )
        w_opt = sol["x"].full().flatten()
        w_opt = self.w(w_opt)

        self.SOC = np.append(self.SOC, w_opt["states", 1, "SOC"])
        self.Pbc = np.append(self.Pbc, w_opt["inputs", 0, "Pbc"])
        self.Pbd = np.append(self.Pbd, w_opt["inputs", 0, "Pbd"])
        self.Pgb = np.append(self.Pgb, w_opt["inputs", 0, "Pgb"])
        self.Pgs = np.append(self.Pgs, w_opt["inputs", 0, "Pgs"])

        return self.get_SOC_opt(), self.get_u_opt()


if __name__ == "__main__":
    ocp = NominelMPC(1, 6)