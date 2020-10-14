from casadi import *
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load

DAYS = 3
SAMPLES_PER_HOUR = 6
T = DAYS * 24  # timehorizon
N = DAYS * 24 * 6  # control intervals

x = MX.sym("x")
u0 = MX.sym("u0")
u1 = MX.sym("u1")
d0 = MX.sym("d0")
d1 = MX.sym("d1")

u = vertcat(u0, u1)
d = vertcat(d0, d1)

pv_values = simulate_pv_cell(max_power=400, days=DAYS, plot=False, add_noise=False)
p_load = simulate_p_load(max_power=300, days=DAYS, plot=False, add_noise=False)

# Constants
C_MAX = 700
n_b = 0.8

# Numerical values
x_inital = 0.3
x_min = 0.3
x_max = 0.9
x_ref = 0.7

u0_max = 1000
u1_max = 500

# ODE
xdot = (1 / C_MAX) * (-n_b * u0)

# Cost parameters
battery_cost = 1
grid_cost = 10
ref_cost = 1

# Objective function
L = battery_cost * (u0 ** 2) + grid_cost * u1 + ref_cost * ((x_ref - x) * 100) ** 2

# Fixed step Runge-Kutta 4 integrator
if True:
    M = 4  # RK4 steps per interval
    DT = T / N / M
    f = Function("f", [x, u, d], [xdot, L])
    X0 = MX.sym("X0")
    U = MX.sym("U", 2)
    D = MX.sym("D", 2)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U, D)
        k2, k2_q = f(X + DT / 2 * k1, U, D)
        k3, k3_q = f(X + DT / 2 * k2, U, D)
        k4, k4_q = f(X + DT * k3, U, D)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    F = Function("F", [X0, U, D], [X, Q], ["x0", "p", "d"], ["xf", "qf"])

# Evaluate at a test point
Fk = F(x0=0.5, p=[500, 0], d=[30, 0])
print(Fk["xf"])
print(Fk["qf"])

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
    Uk = MX.sym("U_" + str(k), 2)
    w += [Uk]
    lbw += [-u0_max, -u1_max]
    ubw += [u0_max, u1_max]
    w0 += [0, 0]

    d0_k = pv_values[k]
    d1_k = p_load[k]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk, d=[d0_k, d1_k])
    Xk_end = Fk["xf"]
    J = J + Fk["qf"]

    # New NLP variable for state at end of interval
    Xk = MX.sym("X_" + str(k + 1))
    w += [Xk]
    lbw += [x_min]
    ubw += [x_max]
    w0 += [0]

    # Add equality constraint
    g += [Xk_end - Xk, Uk[0] + Uk[1] + d0_k - d1_k]
    lbg += [0, 0]
    ubg += [0, 0]

# Create an NLP solver
prob = {"f": J, "x": vertcat(*w), "g": vertcat(*g)}
solver = nlpsol("solver", "ipopt", prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"].full().flatten()


# Plot the solution
import matplotlib.pyplot as plt

plt.figure()
plt.clf()
plt.plot(linspace(0.0, DAYS * 24.0, N), w_opt[1::3], "-")
plt.plot(linspace(0.0, DAYS * 24.0, N), w_opt[2::3], "-")
plt.title("Microgrid - multiple shooting")
plt.xlabel("Time [h]")
plt.legend(["P_Bat", "P_Grid"])
plt.grid()

plt.figure()
plt.plot(linspace(0.0, DAYS * 24.0, N + 1), w_opt[0::3])
plt.xlabel("Time [h]")
plt.ylabel("SOC [%]")
plt.title("State of charge")

plt.show()
