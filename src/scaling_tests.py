#%%
from build_test_cases import (
    eight_node_test_graph,
    smallest_test_graph,
    example_graph,
    multiple_example_graph,
    pybbn_generate_graph,
)
from lazy_learner import LazyLearner
import networkx as nx
import numpy as np

from matplotlib import pyplot as plt
from pybbn.generator.bbngenerator import (
    convert_for_exact_inference,
    generate_singly_bbn,
    generate_multi_bbn,
)
from pybbn.generator.bbngenerator import convert_for_exact_inference

# %%
n = 50
number_of_trials_per_size = 10
x = list(range(1, n + 1))
preprocess_measurements, solver_measurements = [], []
for i in range(20, n + 2):
    print(f"entering {i}-th iteration")
    preprocess_time, solver_time = 0, 0
    for j in range(number_of_trials_per_size):
        np.random.seed(j)
        g, jointree = pybbn_generate_graph(generate_multi_bbn, i, i)
        ll = LazyLearner(g, jointree, random_graph=True)
        # ll.plot_graph()
        random_node = np.random.randint(low=0, high=i)
        preprocess_time_j, solver_time_j = ll.initialize_ip_gurobi(random_node)
        preprocess_time += preprocess_time_j / number_of_trials_per_size
        solver_time += solver_time_j / number_of_trials_per_size
    preprocess_measurements.append(preprocess_time)
    solver_measurements.append(solver_time)


#%%
fig, ax = plt.subplots(1, 2, sharex=True)

for i in range(2):
    if i == 0:
        ax[i].plot(x, preprocess_measurements)
        ax[i].set_ylim(bottom=0)
        ax[i].set_ylabel("Processing Time [s]")
        ax[i].set_xlabel("Number of nodes")
        ax[i].set_title("Preprocessing steps")
    else:
        ax[i].plot(x, solver_measurements)
        ax[i].set_ylim(bottom=0)
        ax[i].set_ylabel("Processing Time [s]")
        ax[i].set_xlabel("Number of nodes")
        ax[i].set_title("IP solver")
fig = plt.gcf()
fig.set_size_inches(12, 8)
