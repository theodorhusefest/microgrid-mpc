from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from pv_cellM import simulate_pv_cell
from p_loadM import simulate_p_load


DAYS = 4
sim_time = DAYS * 24  # Simulation time
T = 1/6  # Sampling time
N = 5  # Time horizon of the MPC

pv_values = simulate_pv_cell(
    resulution=T * 60,
    max_power=80,
    days=DAYS,
    plot=True,
    add_noise=True)
p_load = simulate_p_load(
    resulution=T * 60,
    max_power=60,
    days=DAYS,
    plot=True,
    add_noise=True)

C_max = 700  # Max battery capacity in kWh
eta_bat = 0.8  # Efficiency of the battery charge/discharge
scale = eta_bat/C_max  # Scaling coefficient for later use in the objective func

# The cost/weights in the objective function
battery_cost = 0.1
grid_imp_cost = 0.01
grid_exp_cost = 0.01
ref_cost = 1000000000

# Maximum power of the controllers
bat_max = 1000
grid_max = 500

# A simple model of a battery based system.
SOC = SX.sym('SOC')  # State of charge
states = SOC
n_states = states.shape[0]

# The system has two control inputs, battery charge/discharge and the grid
Pb = SX.sym('Pb')
Pg_exp = SX.sym('Pg_exp')
Pg_imp = SX.sym('Pg_imp')
controls = vertcat(Pb, Pg_exp, Pg_imp)
n_controls = controls.shape[0]

"""
Pv = MX.sym('PV')
Load = MX.sym('Load')
disturbances = vertcat(Pv, Load)
n_disturbance = disturbances.shape[0]
"""
# rhs is the right hand side of the differential equation for the states.
rhs = -(Pb * eta_bat / C_max)

# Creates a function that maps the states through the rhs side of the equation.
f = Function('f', [states, controls], [rhs])

# U is the symbolic matrix for the optimal control for the prediction horizon.
U = SX.sym('U', n_controls, N)
# P is a symbolic parameter vector with the initial states and the reference states.
P = SX.sym('P', 2 * (n_states + N))
# X is a symbolical matrix of the state prediction
X = SX.sym('X', n_states, (N+1))


# Declaring objective function and constraint list to be used later.
obj = 0
g = []

# Setting weighting matrix for the states
Q = np.zeros([1, 1])
Q[0, 0] = ref_cost

# Setting weighting matrix for the controls
R = np.zeros([3, 3])
R[0, 0] = battery_cost
R[1, 1] = grid_exp_cost
R[2, 2] = grid_imp_cost


st = X[:, 0]
g = vertcat(g, st-P[0:n_states])
for k in range(0, N):
    st = X[:, k]
    con = U[:, k]
    # Objective function. Penalizes difference from reference state and use of controllers.
    obj = obj + (st - P[n_states:2 * n_states]).T @ Q @ (st - P[n_states:2 * n_states]) + (con.T/scale) @ R @ (con/scale)
    st_next = X[:, k+1]
    # 4th degree Runga-Kutta
    k1 = f(st, con)
    k2 = f(st + T/2 * k1, con)
    k3 = f(st + T/2 * k2, con)
    k4 = f(st + T * k3, con)
    st_next_RK4 = st + T/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    g = vertcat(g, st_next-st_next_RK4)

# Here we create the constraints.
for k in range(0, N):
    g = vertcat(g, U[0, k] - U[1, k] + U[2, k] + P[n_states*2 + 2 * k] - P[n_states*2 + 2 * k + 1])

# Reshapes the optimal controller U since the solver takes a vector as argument. U is our optimization variables.
OPT_variables = vertcat(reshape(X, n_states * (N+1), 1), reshape(U, n_controls*N, 1))
OPT_length = (N * n_controls + (N+1) * n_states)
# Creates a dict to be used in the solver. function f is our objective function, the variables to be optimized 'x' is
# our control variables, g is our constraints and p is our parameters with initial value and references.
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

# Options for the solver, add to the nlpsol as a last argument. These arguments stop the printing of the solver.
s_opts = {'ipopt': {'print_level': 0}, 'print_time': False}

# Solves our optimization problem.
solver = nlpsol('solver', 'ipopt', nlp_prob, s_opts)

# Constraints on our states, as defined above.
lbg = 0
ubg = 0
"""lbg = [None] * (((N+1) * n_states) + N)
ubg = [None] * (((N+1) * n_states) + N)
for i in range(0, n_states*(N+1)):
    lbg[i] = 0
    ubg[i] = 0"""
# Constraints on the optimization variables.
lbx = [None] * (N * n_controls + (N+1) * n_states)
ubx = [None] * (N * n_controls + (N+1) * n_states)
for i in range(0, (N+1) * n_states):  # Constraints on the states
    lbx[i] = 0.3
    ubx[i] = 0.9

"""
# Constraints on the controllers
for i in range((N + 1) * n_states, (N * n_controls + (N+1) * n_states), 2):
    lbx[i] = -bat_max
    ubx[i] = bat_max
    lbx[i+1] = -grid_max
    ubx[i+1] = grid_max
"""
for i in range((N + 1) * n_states, (N * n_controls + (N+1) * n_states), 3):
    lbx[i] = -bat_max
    ubx[i] = bat_max
    lbx[i + 1] = 0
    ubx[i + 1] = grid_max
    lbx[i + 2] = 0
    ubx[i + 2] = grid_max

print(lbx)
print(ubx)

runs = int(sim_time/T) + 1
xx = np.empty((n_states, runs))
xx1 = np.empty((runs, N+1, n_states))
t = [None] * runs
t0 = 0
# Initial states
x0 = 0.5
# Reference/desired value for the states
xs = 0.7
xx[:, 0] = x0
t[0] = 0

u0 = np.zeros((N, n_controls))
X0 = repmat(x0, 1, N+1)
mpciter = 0
u_cl = []
# Loops runs the MPC
while mpciter < sim_time/T:
    # x0 = x0 + np.random.randint(-10, 10)/1000
    p = vertcat(x0, xs)  # Parameter vector of initial state and the reference
    #For loop that adds the disturbance as a parameter. If the
    for i in range(mpciter, N+mpciter):
        if mpciter+N >= sim_time/T:
            for k in range(0, N):
                p = vertcat(p, pv_values[mpciter])
                p = vertcat(p, p_load[mpciter])
            break
        p = vertcat(p, pv_values[i])
        p = vertcat(p, p_load[mpciter])
    x0_loop = vertcat(reshape(X0.T, n_states * (N+1), 1), reshape(u0.T, n_controls*N, 1))  # control inputs as the optimization variables for the solver.
    sol = solver(x0=x0_loop, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)  # Calculates the optimal control.
    u = reshape(sol['x'].full()[n_states*(N+1):OPT_length+1].T, n_controls, N).T  # Control is set to the new optimal control
    # Makes sure only import or export is activated.
    for i in range(0, N):
        if u[i, 2] - u[i, 1] < 0:
            u[i, 1] = u[i, 1] - u[i, 2]
            u[i, 2] = 0
        elif u[i, 2] - u[i, 1] >= 0:
            u[i, 2] = u[i, 2] - u[i, 1]
            u[i, 1] = 0
    xx1[mpciter] = reshape(sol['x'].full()[0:n_states*(N+1)].T, n_states, N+1).T  # Stores the optimal states.
    u_cl = vertcat(u_cl, u[0, :])  # Stores the the optimal control for the first time step of each horizon.
    t[mpciter] = t0

    st = x0
    con = u[0, :].T
    f_value = f(st, con)
    st = st + (T*f_value)
    x0 = st.full()
    t0 = t0 + T
    u0 = vertcat(u[1:u.shape[0], :], u[u.shape[0]-1, :])

    X0 = reshape(sol['x'].full()[0:n_states*(N+1)].T, n_states, N+1).T
    xx[:, mpciter+1] = x0
    X0 = vertcat(X0[1:N+1, :], X0[N, :])
    mpciter = mpciter + 1

plt.figure(1)
plt.plot(t, xx[0])
plt.figure(2)
plt.plot(t[0:runs-1], u_cl, drawstyle="steps-post")
plt.show()