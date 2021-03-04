start = time.time()
T = 1
N = 18
N_sim = 100
N_scenarios = 12

ocp = ScenarioMPC(T, N, N_scenarios)

pv_base = np.ones(N) * 10
l1_base = np.ones(N) * 50
l2_base = np.ones(N) * 10
E_base = np.ones(N) * 1
s = ocp.scenarios(0)

x_initial = 0.4
step = 40


data = pd.read_csv("./data/data_oct20.csv", parse_dates=["date"]).iloc[::10]
L1 = Load(N, "./data/loads_train.csv", "L1")
L2 = Load(N, "./data/loads_train.csv", "L2")

l1_measurement = L1.mean[step]
l2_measurement = L2.mean[step]

l1_scenarios = get_monte_carlo_scenarios(
    l1_measurement, step, N, N_sim, N_scenarios, L1, L1.scaled_mean_pred, data
)

l2_scenarios = get_monte_carlo_scenarios(
    l2_measurement, step, N, N_sim, N_scenarios, L2, L2.scaled_mean_pred, data
)

for i in range(N_scenarios):
    for k in range(N):
        s["scenario" + str(i), "data", k, "pv"] = pv_base[k] + np.random.normal(
            0, 15, 1
        )
        s["scenario" + str(i), "data", k, "l1"] = l1_scenarios[i][k]
        s["scenario" + str(i), "data", k, "l1"] = l2_scenarios[i][k]

        s["scenario" + str(i), "data", k, "E"] = E_base[k]


SOC, Pbc, Pbd, Pgb, Pgs = ocp.solve_scenario_tree(0.4, s)


def plot_serie(N, N_scenarios, serie, title):
    plt.figure()
    for k in range(N_scenarios):
        plt.plot(range(N), serie[k], label="Scenario" + str(k))
    plt.legend()
    plt.title(title)


plot_serie(N, N_scenarios, SOC, "SOC")
plot_serie(N - 1, N_scenarios, Pbc, "Pbc")
plot_serie(N - 1, N_scenarios, Pbd, "Pbd")
plot_serie(N - 1, N_scenarios, Pgb, "Pgb")
plot_serie(N - 1, N_scenarios, Pgs, "Pgs")

plt.show()


print("\n Used {}s.".format(time.time() - start))
