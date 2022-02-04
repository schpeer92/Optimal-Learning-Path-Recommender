#%%
from build_test_cases import eight_node_test_graph, smallest_test_graph,example_graph, multiple_example_graph, pybbn_generate_graph
from lazy_learner import LazyLearner
import networkx as nx
import numpy as np

from matplotlib import pyplot as plt
from pybbn.generator.bbngenerator import convert_for_exact_inference, generate_singly_bbn,generate_multi_bbn
from pybbn.generator.bbngenerator import convert_for_exact_inference

#testing different graph instances and scaling them 
#%%
g,jointree = smallest_test_graph()
ll = LazyLearner(g,jointree)
ll.plot_graph()
#ll.initialize_ip(2,0.5)

#%%
g,jointree = eight_node_test_graph()
ll = LazyLearner(g,jointree)
ll.plot_graph()
#ll.initialize_ip(7,0.5)
#%%
n = 2
g,jointree=multiple_example_graph(n)
ll = LazyLearner(g,jointree)
ll.plot_graph()
# %%
g,jointree = example_graph()
ll = LazyLearner(g,jointree)
ll.plot_graph()
ll.initialize_ip(3,0.5)

#%%
n = 3
x = list(range(1*4,n*4,4))
solver_times = []
total_times = []

for i in range(1,n):
    print(f"entered {i}-th iteration")
    g,jointree = multiple_example_graph(i)
    ll = LazyLearner(g,jointree)
    total_time, solver_time = ll.initialize_ip(i*4-1,0.5)
    solver_times.append(solver_time)
    total_times.append(total_time)

fig, ax = plt.subplots(1,2, sharex=True)

for i in range(2):
    if i ==0:
        ax[i].plot(x,total_times)
        ax[i].set_ylim(bottom=0)
        ax[i].set_ylabel("Processing Time [s]")
        ax[i].set_xlabel("Number of nodes")
        ax[i].set_title("Preprocessing steps")
    else:
        ax[i].plot(x,solver_times)
        ax[i].set_ylim(bottom=0)
        ax[i].set_ylabel("Processing Time [s]")
        ax[i].set_xlabel("Number of nodes")
        ax[i].set_title("IP solver")
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig("../computational_time_testing.jpg")
# %%
