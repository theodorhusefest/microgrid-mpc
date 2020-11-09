from system import get_integrator


def simulate_SOC(x, u_opt, PV, PV, PV_pred, PL_pred):
    uk = get_real_u(u_opt, PV, PL, PV_pred, PL_pred)

    F = get_integrator(
        1,
        1,
        x,
        u,
        C_MAX=C_MAX,
        nb_c=nb_c,
        nb_d=nb_d,
    )
    Fk = F(x0=x_inital, p=uk)

    x_sim = Fk["xf"].full().flatten()[-1]
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