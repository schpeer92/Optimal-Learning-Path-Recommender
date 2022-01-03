#%%
from build_test_cases import eight_node_test_graph, smallest_test_graph,example_graph, multiple_example_graph, pybbn_generate_graph
from lazy_learner import LazyLearner
import networkx as nx
import numpy as np

from matplotlib import pyplot as plt
from pybbn.generator.bbngenerator import convert_for_exact_inference, generate_singly_bbn,generate_multi_bbn
from pybbn.generator.bbngenerator import convert_for_exact_inference

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

# %%
g,jointree = example_graph()
ll = LazyLearner(g,jointree)
ll.plot_graph()
# %%
ll.initialize_ip(3,0.5)

#%%
n = 8
g,jointree=multiple_example_graph(n)
ll = LazyLearner(g,jointree)
ll.initialize_ip(n*4-1,0.5)

#%%
n = 5
x = list(range(1,n))
solver_times = []
total_times = []

for i in range(1,n):
    print(f"entered {i}-th iteration")
    g,jointree = multiple_example_graph(i)
    ll = LazyLearner(g,jointree)
    total_time, solver_time = ll.initialize_ip(i*4-1,0.5)
    solver_times.append(solver_time)
    total_times.append(total_time)
fig, ax = plt.subplots(1,2)
ax[0].plot(x,total_times)
ax[1].plot(x,solver_times)
ax[0].set_ylim(bottom = 0)
ax[1].set_ylim(bottom = 0)
# %%
n = 20
x = list(range(1,n))
solver_times = []
total_times = []

for i in range(1,n):
    print(f"entered {i}-th iteration")
    g,jointree = multiple_example_graph(i)
    ll = LazyLearner(g,jointree)
    total_time, solver_time = ll.initialize_ip(i*4-1,0.5)
    solver_times.append(solver_time)
    total_times.append(total_time)
fig, ax = plt.subplots(1,2)
ax[0].plot(x,total_times)
ax[1].plot(x,solver_times)
ax[0].set_ylim(bottom = 0)
ax[1].set_ylim(bottom = 0)

# %%
x = list(range(1,13))
fig, ax = plt.subplots(1,2)
ax[0].plot(x,total_times)
ax[1].plot(x,solver_times)
ax[0].set_ylim(bottom = 0)
ax[1].set_ylim(bottom = 0)
# %%
np.random.seed(1)
g,jointree = pybbn_generate_graph(generate_multi_bbn,10,1)
ll = LazyLearner(g,jointree)
ll.plot_graph()
# %%
np.random.randint(200)
# %%
n = 10
fig, ax = plt.subplots(1,2)

ax[0].set_ylim(bottom = 0)
ax[1].set_ylim(bottom = 0)

for number_of_nodes in range(1,n+1):
    number_of_observations = 0
    avg_solver_time = []
    avg_total_time = []
    while number_of_observations < 10:
        random_max_iter = np.random.randint(50)
        g,jointree = pybbn_generate_graph(generate_multi_bbn,number_of_nodes,random_max_iter)
        ll = LazyLearner(g,jointree)
        max_node = max(list(ll.G.nodes))
        try:
            total_time, solver_time = ll.initialize_ip(max_node,0.5)
        except:
            next
        number_of_observations +=1
    avg_solver_time = np.mean(avg_solver_time)
    avg_total_time = np.mean(avg_total_time)
    ax[0].scatter(number_of_nodes,avg_solver_time)
    ax[1].scatter(number_of_nodes,avg_total_time)

# %%
g,jointree = pybbn_generate_graph(generate_multi_bbn, 5,5)
ll = LazyLearner(g,jointree)
ll.initialize_ip(4,0.5)
# %%
ll.join_tree.get_posteriors()
#%%
g,p = generate_multi_bbn(5,5)
bbn = convert_for_exact_inference(g,p)
# %%
bbn.nodes
# %%
from pybbn.pptc.inferencecontroller import InferenceController


# %%
join_tree = InferenceController.apply(bbn)
# %%
join_tree.nodes
# %%
join_tree.get_bbn_node(0)
# %%
join_tree.get_posteriors()
# %%
join_tree.get_bbn_node(0).
# %%
