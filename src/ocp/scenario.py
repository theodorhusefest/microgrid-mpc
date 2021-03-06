from casadi import *
from casadi.tools import *
from utils.helpers import parse_config
import matplotlib.pyplot as plt


class ScenarioOCP:
    def __init__(self, T, N, N_scenarios):

        self.T = T
        self.N = N
        self.N_scenarios = N_scenarios
        conf = parse_config()
        conf_system = conf["system"]

        # Get system constants
        self.C_MAX = conf["battery"]["C_MAX"]
        self.nb_c = conf["battery"]["nb_c"]
        self.nb_d = conf["battery"]["nb_d"]
        self.x_min = conf_system["x_min"]
        self.x_max = conf_system["x_max"]
        self.x_ref = conf_system["x_ref"]
        self.Pb_max = conf_system["Pb_max"]
        self.Pg_max = conf_system["Pg_max"]
        self.battery_cost = conf_system["battery_cost"]
        self.peak_cost = conf_system["peak_cost"]
        self.old_grid_fee = conf_system["old_grid_fee"]
        self.terminal_cost = conf_system["terminal_cost"]
        self.verbose = conf_system["verbose"]

        self.states = struct_symSX([entry("SOC"), entry("Pgb_p")])
        self.inputs = struct_symSX(
            [
                entry("Pbc"),
                entry("Pbd"),
                entry("Pgb"),
                entry("Pgs"),
            ]
        )
        self.slacks = struct_symSX(
            [
                entry("us"),
                entry("ls"),
                entry("s1"),
                entry("s2"),
            ]
        )

        self.w = struct_symSX(
            [
                entry("states", struct=self.states, repeat=self.N),
                entry("inputs", struct=self.inputs, repeat=self.N - 1),
            ]
        )
        self.data = struct_symSX(
            [
                entry("pv"),
                entry("l"),
                entry("E"),
                entry("prob"),
            ]
        )
        self.all_data = struct_symSX([entry("data", struct=self.data, repeat=self.N)])

        scenarios = []
        s_data = []

        for k in range(self.N_scenarios):
            scenarios.append(entry("scenario" + str(k), struct=self.w))
            s_data.append(entry("scenario" + str(k), struct=self.all_data))

        self.s = struct_symSX(scenarios)
        self.s_data = struct_symSX(s_data)

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
            (self.nb_c * self.inputs["Pbc"]) - (self.inputs["Pbd"] / self.nb_d)
        )

    def build_integrator(self):
        """
        Creates the integrator for the current system.
        """

        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function(
            "f",
            [self.states, self.inputs],
            [self.ode],
        )
        X0 = SX.sym("X0")
        U = SX.sym("U", 4)
        X = X0
        for _ in range(M):
            k1 = f(X, U)
            k2 = f(X + DT / 2 * k1, U)
            k3 = f(X + DT / 2 * k2, U)
            k4 = f(X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Function("F", [X0, U], [X], ["x0", "p"], ["xf"])

    def add_stage_cost(self, scenario, k):
        J_scen = self.battery_cost * (
            self.s[scenario, "inputs", k, "Pbc"] + self.s[scenario, "inputs", k, "Pbd"]
        )

        J_scen += self.s_data[scenario, "data", k, "E"] * (
            self.s[scenario, "inputs", k, "Pgb"]
            - 0.9 * self.s[scenario, "inputs", k, "Pgs"]
        )
        J_scen += self.old_grid_fee * self.s[scenario, "inputs", k, "Pgb"]

        J_scen += self.peak_cost * (
            self.s[scenario, "states", k + 1, "Pgb_p"]
            - self.s[scenario, "states", k, "Pgb_p"]
        )
        return J_scen

    def build_scenario_ocp(self, root=None):

        J = 0
        s0 = self.s(0)
        lbs = self.s(0)
        ubs = self.s(0)
        g = []
        lbg = []
        ubg = []

        F = self.build_integrator()

        for j in range(self.N_scenarios):
            J_scen = 0
            scenario = "scenario" + str(j)

            # Set state and inputs constraints
            lbs[scenario, "states", :, "SOC"] = self.x_min
            ubs[scenario, "states", :, "SOC"] = self.x_max
            ubs[scenario, "states", :, "Pgb_p"] = self.Pg_max
            ubs[scenario, "inputs", :, "Pbc"] = self.Pb_max
            ubs[scenario, "inputs", :, "Pbd"] = self.Pb_max
            ubs[scenario, "inputs", :, "Pgs"] = self.Pg_max
            ubs[scenario, "inputs", :, "Pgb"] = self.Pg_max

            for k in range(self.N - 1):
                states_k = self.s[scenario, "states", k, "SOC"]
                inputs_k = self.s[scenario, "inputs", k]

                Fk = F(x0=states_k, p=inputs_k)
                Xk_end = Fk["xf"]

                J_scen += self.add_stage_cost(scenario, k)

                eq_con = [
                    Xk_end - self.s[scenario, "states", k + 1, "SOC"],
                    -self.s[scenario, "inputs", k, "Pbc"]
                    + self.s[scenario, "inputs", k, "Pbd"]
                    + self.s[scenario, "inputs", k, "Pgb"]
                    - self.s[scenario, "inputs", k, "Pgs"]
                    + self.s_data[scenario, "data", k, "pv"]
                    - self.s_data[scenario, "data", k, "l"],
                ]

                ineq_con = [
                    self.s[scenario, "states", k + 1, "Pgb_p"]
                    - self.s[scenario, "states", k, "Pgb_p"],
                    self.s[scenario, "states", k + 1, "Pgb_p"]
                    - self.s[scenario, "inputs", k, "Pgb"],
                ]
                g += eq_con
                g += ineq_con
                lbg += [0] * len(eq_con) + [0, 0]
                ubg += [0] * len(eq_con) + [self.Pg_max] * 2

            # Add terminal cost
            J_scen += self.terminal_cost * self.s[scenario, "states", self.N - 1, "SOC"]

            J += self.s_data[scenario, "data", 0, "prob"] * J_scen

        # Non-anticipativity constraints
        for j in range(self.N_scenarios):
            scenario = "scenario" + str(j)
            for i in range(j + 1, self.N_scenarios):
                subscenario = "scenario" + str(i)

                if scenario == subscenario:
                    continue
                g_ant = [
                    self.s[scenario, "inputs", 0, "Pbc"]
                    - self.s[subscenario, "inputs", 0, "Pbc"],
                    self.s[scenario, "inputs", 0, "Pbd"]
                    - self.s[subscenario, "inputs", 0, "Pbd"],
                ]
                g += g_ant
                lbg += [0] * len(g_ant)
                ubg += [0] * len(g_ant)

        prob = {"f": J, "x": self.s, "g": vertcat(*g), "p": self.s_data}
        if self.verbose:
            self.solver = nlpsol("solver", "ipopt", prob)
        else:
            opts = {
                "verbose_init": False,
                "ipopt": {"print_level": 2, "sb": "yes"},
                "print_time": False,
            }
            self.solver = nlpsol("solver", "ipopt", prob, opts)

        return [s0, lbs, ubs, lbg, ubg]

    def solve_nlp(self, params, data, argmax):
        # Solve the NLP
        sol = self.solver(
            x0=params[0],
            lbx=params[1],
            ubx=params[2],
            lbg=params[3],
            ubg=params[4],
            p=data,
        )
        s_opt = sol["x"].full().flatten()
        s_opt = self.s(s_opt)

        i = int(np.floor(self.N_scenarios / 2))
        # i = argmax
        self.SOC = np.append(self.SOC, s_opt["scenario" + str(i), "states", 1, "SOC"])
        self.Pbc = np.append(self.Pbc, s_opt["scenario" + str(i), "inputs", 0, "Pbc"])
        self.Pbd = np.append(self.Pbd, s_opt["scenario" + str(i), "inputs", 0, "Pbd"])
        self.Pgb = np.append(self.Pgb, s_opt["scenario" + str(i), "inputs", 0, "Pgb"])
        self.Pgs = np.append(self.Pgs, s_opt["scenario" + str(i), "inputs", 0, "Pgs"])

        return self.get_SOC_opt(), self.get_u_opt()
