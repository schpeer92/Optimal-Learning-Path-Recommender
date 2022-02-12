#%%
from build_test_cases import multiple_example_graph
from lazy_learner import LazyLearner
from matplotlib import pyplot as plt


#%%
n = 25
x = list(range(1 * 4, n * 4, 4))

for solver in ["gurobi", "gekko"]:
    solver_times = []
    total_times = []
    for i in range(1, n):
        print(f"entered {i}-th iteration")
        g, jointree = multiple_example_graph(i)
        ll = LazyLearner(g, jointree)
        if solver == "gekko":
            total_time, solver_time = ll.initialize_ip_gekko(i * 4 - 1)
        else:
            total_time, solver_time = ll.initialize_ip_gurobi(i * 4 - 1)
        solver_times.append(solver_time)
        total_times.append(total_time)

    fig, ax = plt.subplots(1, 2, sharex=True)

    for i in range(2):
        if i == 0:
            ax[i].plot(x, total_times)
            ax[i].set_ylim(bottom=0)
            ax[i].set_ylabel("Processing Time [s]")
            ax[i].set_xlabel("Number of nodes")
            ax[i].set_title("Preprocessing steps")
        else:
            ax[i].plot(x, solver_times)
            ax[i].set_ylim(bottom=0)
            ax[i].set_ylabel("Processing Time [s]")
            ax[i].set_xlabel("Number of nodes")
            ax[i].set_title("IP solver")
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.title(f"Computational Time for {solver}")
    plt.savefig(f"../computational_time_testing_{solver}.jpg")
