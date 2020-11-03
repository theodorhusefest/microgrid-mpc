import time
import numpy as np
import matplotlib.pyplot as plt

from casadi import MX, nlpsol, vertcat
from utils.plots import plot_SOC, plot_control_actions
from system import get_integrator


def solve_optimization(
    x_inital,
    uk_1,
    T,
    N,
    PV,
    PL,
    PV_pred,
    PL_pred,
    C_MAX=700,
    nb_c=0.8,
    nb_d=0.8,
    x_min=0.3,
    x_max=0.9,
    x_ref=0.7,
    Pb_max=1000,
    Pg_max=500,
    battery_cost=1,
    grid_cost=1,
    grid_buy=1,
    grid_sell=1,
    ref_cost=0.1,
    verbose=False,
    openloop=True,
):
    """
    Solves the open loop optimization problem starting at x_inital,
    and till the end of the period.

    x: State of charge

    u0 = P_bat charge
    u1 = P_bat discharge
    u2 = P_grid buy
    u3 = P_grid sell

    d1 = P_PV
    d2 = P_L

    """

    # Define symbolic varibales
    x = MX.sym("x")

    u0 = MX.sym("u0")
    u1 = MX.sym("u1")
    u2 = MX.sym("u2")
    u3 = MX.sym("u3")

    u = vertcat(u0, u1, u2, u3)

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
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym("U_" + str(k), 4)
        w += [Uk]
        lbw += [0, 0, 0, 0]
        ubw += [Pb_max, Pb_max, Pg_max, Pg_max]
        w0 += [0, 0, 0, 0]

        d0_k = PV_pred[k]
        d1_k = PL_pred[k]

        # Integrate till the end of the interval
        F = get_integrator(
            T,
            N,
            x,
            u,
            uk_1,
            x_ref=x_ref,
            battery_cost=battery_cost,
            grid_buy=grid_buy[k],
            grid_sell=grid_sell[k],
            ref_cost=ref_cost,
            C_MAX=C_MAX,
            nb_c=nb_c,
            nb_d=nb_d,
        )

        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk["xf"]
        J = J + Fk["qf"]

        # New NLP variable for state at end of interval
        Xk = MX.sym("X_" + str(k + 1))
        w += [Xk]
        lbw += [x_min]
        ubw += [x_max]
        w0 += [0]

        # Add equality constraints
        g += [
            Xk_end - Xk,
            -Uk[0] + Uk[1] + Uk[2] - Uk[3] + d0_k - d1_k,
            Uk[0] * Uk[1],
            Uk[2] * Uk[3],
        ]

        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Create an NLP solver
    prob = {"f": J, "x": vertcat(*w), "g": vertcat(*g)}
    if verbose:
        solver = nlpsol("solver", "ipopt", prob)
    else:
        opts = {"verbose_init": True, "ipopt": {"print_level": 0}, "print_time": False}
        solver = nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    x_opt = w_opt[0::5]
    u_opt = [w_opt[1::5], w_opt[2::5], w_opt[3::5], w_opt[4::5]]

    if False:
        uk_1 = np.asarray([u_[0] for u_ in u_opt])
        return x_opt[1], uk_1, x_opt, u_opt

    uk = get_real_u(u_opt, PV, PL, PV_pred, PL_pred)
    F = get_integrator(
        1,
        1,
        x,
        u,
        uk_1,
        C_MAX=C_MAX,
        nb_c=nb_c,
        nb_d=nb_d,
    )
    Fk = F(x0=x_inital, p=uk)

    x_sim = Fk["xf"].full().flatten()[-1]
    uk_1 = np.copy(uk)
    return x_sim, uk, x_opt, u_opt


def get_real_u(u_opt, PV, PL, PV_pred, PL_pred):
    """
    Calculates the real inputs when there are errors between
    prediction and real PV and load values
    """
    u = np.asarray([u_[0] for u_ in u_opt])
    e_PV = PV[0] - PV_pred[0]
    e_PL = PL[0] - PL_pred[0]
    e_Pbat = e_PV - e_PL
    if e_Pbat > 0:
        u[0] += e_Pbat
    else:
        u[1] -= e_Pbat
    return u