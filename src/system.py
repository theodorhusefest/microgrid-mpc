from casadi import Function, MX
from utils.helpers import parse_config


def get_objective_function(
    x,
    u,
    uk_1,
    x_ref=0.7,
    battery_cost=1,
    grid_cost=1,
    grid_buy=1,
    grid_sell=1,
    ref_cost=1,
):
    """
    Returns the objective function
    """
    L = (
        battery_cost * (u[0] + u[1])
        # + 5 * (u[0] - uk_1[0]) ** 2
        # + 5 * (u[1] - uk_1[1]) ** 2
        + 10 * grid_buy * u[2]
        - 10 * grid_sell * 0.7 * u[3]
        + grid_cost * (u[2] + u[3]) ** 2
        + ref_cost * ((x_ref - x) * 100) ** 2
    )
    return L


def get_ode(x, u0, u1, C_MAX=700, nb_c=0.8, nb_d=0.8):
    """
    Returns the objective function.
    Can be dynamically updated
    """
    xdot = (1 / C_MAX) * ((nb_c * u0) - (u1 / nb_d))
    return xdot


def get_integrator(
    T,
    N,
    x,
    u,
    uk_1,
    x_ref=0.8,
    battery_cost=1,
    grid_cost=1,
    grid_buy=1,
    grid_sell=1,
    ref_cost=1,
    C_MAX=700,
    nb_c=0.8,
    nb_d=0.8,
):
    """
    Creates the given integrator for the current system.
    """
    xdot = get_ode(x, u[0], u[1], C_MAX=C_MAX, nb_c=nb_c, nb_d=nb_d)
    L = get_objective_function(
        x,
        u,
        uk_1,
        x_ref=x_ref,
        battery_cost=battery_cost,
        grid_cost=grid_cost,
        grid_buy=grid_buy,
        grid_sell=grid_sell,
        ref_cost=ref_cost,
    )

    M = 4  # RK4 steps per interval
    DT = T / N / M
    f = Function("f", [x, u], [xdot, L])
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
