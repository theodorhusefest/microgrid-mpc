from casadi import *
from simulations.pv_cell import simulate_pv_cell
from simulations.p_load import simulate_p_load


def open_loop_optimization(
    x_inital,
    T,
    N,
    pv_values,
    p_load,
    C_MAX= 700, 
    nb_c= 0.8,
    nb_d= 0.8,
    x_min= 0.3,
    x_max= 0.9,
    x_ref= 0.7,
    Pb_max= 1000,
    Pg_max= 500,
    battery_cost= 1,
    grid_buy= 10000,
    grid_sell= 10,
    ref_cost= 0.01,
    verbose= False,
    plot= True):

    """ 
    Solves the open loop optimization problem starting at x_inital, 
    and till the end of the period.
    """

    # Define symbolic varibales
    x = MX.sym('x')

    Pb_charge = MX.sym('u0')
    Pb_discharge = MX.sym('u1')
    Pg_buy = MX.sym('u2')
    Pg_sell = MX.sym('u3')

    d0 = MX.sym('d0')
    d1 = MX.sym('d1')

    u = vertcat(Pb_charge, Pb_discharge, Pg_buy, Pg_sell)
    d = vertcat(d0, d1)

    # ODE
    xdot = (1/C_MAX)*((nb_c*Pb_charge) - (nb_d*Pb_discharge))

    # Objective function
    L =  battery_cost*(Pb_charge + Pb_discharge) + grid_buy*Pg_buy - grid_sell*Pg_sell + ref_cost*((x_ref-x)*100)**2

    # Fixed step Runge-Kutta 4 integrator
    if True:
        M = 4 # RK4 steps per interval
        DT = T/N/M
        f = Function('f', [x, u, d], [xdot, L])
        X0 = MX.sym('X0')
        U = MX.sym('U', 4)
        D = MX.sym('D', 2)
        X = X0
        Q = 0
        for j in range(M):
            k1, k1_q = f(X, U, D)
            k2, k2_q = f(X + DT/2 * k1, U, D)
            k3, k3_q = f(X + DT/2 * k2, U, D)
            k4, k4_q = f(X + DT * k3, U, D)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        F = Function('F', [X0, U, D], [X, Q],['x0','p','d'],['xf','qf'])

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    Xk = MX.sym('X0')
    w += [Xk]
    lbw += [x_inital]
    ubw += [x_inital]
    w0 += [x_inital]

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), 4)
        w   += [Uk]
        lbw += [0, 0, 0, 0]
        ubw += [Pb_max, Pb_max, Pg_max, Pg_max]
        w0  += [0, 0, 0, 0]

        d0_k = pv_values[k]
        d1_k = p_load[k]

        # Integrate till the end of the interval
        Fk = F(x0=Xk,p=Uk, d=[d0_k, d1_k])
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1))
        w   += [Xk]
        lbw += [x_min]
        ubw += [x_max]
        w0  += [0]

        # Add equality constraints
        g   += [
            Xk_end-Xk,
            Uk[0] - Uk[1] + Uk[2] - Uk[3] + d0_k - d1_k,
            Uk[0]*Uk[1],
            Uk[2]*Uk[3]]

        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    opts = {"verbose_init": True, "ipopt": {"print_level": 1}, "print_time": True}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    x_opt = w_opt[0::5]
    u0_opt = w_opt[1::5]
    u1_opt = w_opt[2::5]
    u2_opt = w_opt[3::5]
    u3_opt = w_opt[4::5]

    if plot == True:
        # Plot the solution
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(linspace(0., T, N), u0_opt,'-')
        plt.plot(linspace(0., T, N), -u1_opt,'-')
        plt.plot(linspace(0., T, N), u2_opt,'-')
        plt.plot(linspace(0., T, N), -u3_opt,'-')
        plt.title('Microgrid - multiple shooting')
        plt.xlabel('Time [h]')
        plt.legend(['Pb_charge','Pb_discharge', 'Pg_buy', 'Pg_sell'])
        plt.grid()

        plt.figure()
        plt.plot(linspace(0., T, N+1), x_opt)
        plt.xlabel('Time [h]')
        plt.ylabel('SOC [%]')
        plt.title('State of charge')

        plt.show()

    return  x_opt, [u0_opt, u1_opt, u2_opt, u3_opt] 



if __name__ == "__main__":
    x_inital = 0.4
    T = 24
    N= 24*6

    pv_values = simulate_pv_cell(
        max_power=400,
        days=1,
        plot=False,
        add_noise=False)
    p_load = simulate_p_load(
        max_power=300,
        days=1,
        plot=False, 
        add_noise=False)
    
    open_loop_optimization(
        x_inital, T, N, 
        pv_values, p_load)