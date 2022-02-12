import enum
from platform import node
import queue

from gekko import GEKKO
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations, product
import numpy as np
from pybbn.graph.jointree import EvidenceBuilder
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt
import time
from numba import jit


import time
import functools


def timer(func):
    """
    source:https://stackoverflow.com/questions/5478351/python-time-measure-function
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs):

        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure


class LazyLearner:
    def __init__(
        self,
        nx_graph,
        join_tree,
        random_graph=False,
    ) -> None:
        self.G = nx_graph
        self.join_tree = join_tree
        self.random = random_graph

    def plot_graph(self):
        my_pos = graphviz_layout(self.G, prog="dot")
        nx.draw(
            self.G,
            pos=my_pos,
            with_labels=True,
            node_color="orange",
            node_size=400,
            edge_color="black",
            linewidths=1,
            font_size=15,
        )
        plt.show()

    def build_evidence(self, node, value):
        """Instantiation of parent nodes.

        Args:
            node : parent node
            value : 0 or 1
        """
        state = "state1" if value == 1 else "state0"
        ev = (
            EvidenceBuilder()
            .with_node(self.join_tree.get_bbn_node(node))
            .with_evidence(state, value)
            .build()
        )
        self.join_tree.set_observation(ev)

    def posterior_proba(
        self, parent_node_states: tuple, permutation_to_node_map: tuple, node
    ) -> float:
        """Instantiates the states of parents and returns the probability of target node

        Args:
            parent_node_states (tuple): realization of parents
            permutation_to_node_map (tuple): maps a realization to their parent node
            node ([type]): target node from which we want to get the probability given its parents

        Returns:
            float: probability of a node of active state given its parents
        """
        for binary_value, parent_node in zip(
            parent_node_states, permutation_to_node_map
        ):
            self.build_evidence(parent_node, binary_value)
        if self.random == False:
            proba = self.join_tree.get_posteriors()[node]
        else:
            proba = self.join_tree.get_posteriors()[str(node)]

        self.join_tree.unobserve_all()
        return proba["state1"]

    def create_CPT_gekko(self, node: int) -> np.array:
        """Create CPT tables for all nodes"""

        parents = tuple(self.G.predecessors(node))
        return (
            self.model.Array(
                self.model.Const, 1, value=self.posterior_proba((), (), node)
            )
            if not parents
            else self.create_CPT_with_parents_gekko(node)
        )

    def parent_permutations(self, node) -> tuple:
        parents = tuple(self.G.predecessors(node))
        num_parents = len(parents)
        permutations = tuple(product(range(2), repeat=num_parents))

        return permutations, parents

    def create_CPT_with_parents_gekko(self, node):
        permutations, parents = self.parent_permutations(node)
        num_parents = len(parents)
        dimensions = tuple(2 for _ in range(num_parents))
        cpt = self.model.Array(self.model.Const, dimensions)

        for permutation in permutations:
            cpt[permutation] = self.posterior_proba(permutation, parents, node)
        return cpt

    def backwards_traversal(self, target_node: int) -> list:
        queue = list(self.G.predecessors(target_node))
        for node in queue:
            queue += list(self.G.predecessors(node))
        return list(set(queue))

    def create_CPT_for_all_nodes_gekko(self, Q, target_node):
        return {node: self.create_CPT_gekko(node) for node in Q + [target_node]}

    def create_binary_variables_gekko(self, Q):
        return {node: self.model.Var(lb=0, ub=1, integer=True, value=1) for node in Q}

    def model_permutation_equations_gekko(
        self, cpt, x, realization, parents, target_node_value, threshold: float = 0.5
    ):
        """loop over each realization of a nodes parents and check whether this is taken
        by our decision variables or not. If it is taken check if the CPT of that realization
        yields a probability higher than threshold.
        """
        if parents == ():
            self.model.Equation(cpt[0] >= threshold * x[target_node_value])
        elif target_node_value == 1:
            is_taken = 1
            for instant_state, parent in zip(realization, parents):
                is_taken *= 1 - x[parent] if instant_state == 0 else x[parent]
            self.model.Equation(cpt[realization] >= is_taken * threshold)
        else:
            is_taken = 1
            for instant_state, parent in zip(realization, parents):
                is_taken *= 1 - x[parent] if instant_state == 0 else x[parent]
            self.model.Equation(
                cpt[realization] >= is_taken * threshold * x[target_node_value]
            )

    def model_permutations_gekko(self, node, x, cpt, threshold, target_node=False):
        permutations, parents = self.parent_permutations(node)
        for realization in permutations:
            if target_node:
                # 1 argument to enforce the threhshold condition
                self.model_permutation_equations_gekko(
                    cpt, x, realization, parents, 1, threshold
                )
            else:
                self.model_permutation_equations_gekko(
                    cpt, x, realization, parents, node, threshold
                )

    def initialize_ip_gekko(self, target_node, threshold=0.5):

        start = time.time()
        self.model = GEKKO(remote=False)
        threshold = self.model.Const(threshold)
        Q = self.backwards_traversal(target_node)
        CPT = self.create_CPT_for_all_nodes_gekko(Q, target_node)
        x = self.create_binary_variables_gekko(Q)

        # model IP for target node
        for node in Q:
            self.model_permutations_gekko(
                node, x, CPT[node], threshold, target_node=False
            )

        # model IP for all pare nodes (if taken property)
        self.model_permutations_gekko(
            target_node, x, CPT[target_node], threshold, target_node=True
        )

        # only optimizing parent path - other might yield problems
        _, target_parents = self.parent_permutations(target_node)
        y = self.model.Array(self.model.Var, 2 ** len(target_parents))
        z = self.model.Var()

        target_parents_variables = [parent for parent in Q if parent in target_parents]
        self.model.Equation(
            z == sum(2**i * x[xi] for i, xi in enumerate(target_parents_variables))
        )

        for i, yi in enumerate(y):
            # if i==z then the equation is 1 0 otherwise
            # quadratic - no division by 0
            self.model.Equation(yi == 0 ** ((i - z) ** 2))

        self.model.Minimize(
            self.model.sum([x[i] for i in x]) - CPT[target_node].flatten() @ y
        )
        self.model.options.SOLVER = 1

        end = time.time()
        total_time = end - start
        self.model.solve(disp=False)

        solver_time = self.model.options.SOLVETIME
        print(
            f"Time taken to solve instance:{solver_time}\nTotal solving time:{total_time}\nOptimal path: {x}"
        )
        return total_time, solver_time

    def create_CPT_gp(self, node: int) -> np.array:
        """Create CPT tables for all nodes"""
        parents = tuple(self.G.predecessors(node))
        return (
            np.array(self.posterior_proba((), (), node))
            if not parents
            else self.create_CPT_with_parents_gp(node)
        )

    def create_CPT_with_parents_gp(self, node):

        permutations, parents = self.parent_permutations(node)
        num_parents = len(parents)
        dimensions = tuple(2 for _ in range(num_parents))
        cpt = np.zeros(dimensions)

        for permutation in permutations:
            cpt[permutation] = self.posterior_proba(permutation, parents, node)

        return cpt

    def create_CPT_for_all_nodes_gp(self, Q, target_node):
        return {node: self.create_CPT_gp(node) for node in Q + [target_node]}

    def model_CPT_constraints(self, Q, Qx_map, CPT, Y, x, threshold):
        """Check if the decision variable yields a probability higher than threshold if taken"""
        for node in Q:
            self.model.addConstr(
                gp.quicksum(
                    CPT[node].flatten()[i] * Y[node][i]
                    for i in range(len(CPT[node].flatten()))
                )
                >= x[Qx_map[node]] * threshold
            )

    def model_decision_variable_constraints(self, Q, Y, x):
        """if node is taken and active than we have to choose a parent realization"""
        self.model.addConstrs(gp.quicksum(Y[node]) == x[i] for i, node in enumerate(Q))

    def model_permutations_gp(self, Q, Qx_map, Y, x):
        """force nodes to be active if a realization has been chosen"""
        for node in Q:
            permutations, parents = self.parent_permutations(node)
            for i, realization in enumerate(permutations):
                for j, parent in enumerate(parents):
                    self.model.addConstr(
                        realization[j] * Y[node][i] <= x[Qx_map[parent]]
                    )

    def initialize_ip_gurobi(self, target_node, threshold=0.5):
        start = time.time()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        self.model = gp.Model(env=env)
        Q = self.backwards_traversal(target_node)

        CPT = self.create_CPT_for_all_nodes_gp(Q, target_node)
        x = self.model.addMVar(shape=len(Q), vtype=GRB.BINARY)

        # mapping Q to x, because x is an array
        Qx_map = {node: i for i, node in enumerate(Q)}
        Y = {
            node: self.model.addMVar(
                shape=2 ** len(tuple(self.G.predecessors(node))), vtype=GRB.BINARY
            )
            for node in Q + [target_node]
        }
        # model threshold constraints
        self.model_CPT_constraints(Q, Qx_map, CPT, Y, x, threshold)
        self.model.addConstr(CPT[target_node].flatten() @ Y[target_node] >= threshold)
        self.model_decision_variable_constraints(Q, Y, x)
        # model decision variables
        self.model.addConstr(Y[target_node].sum() == 1)
        # force x to be 1 if realization is taken
        self.model_permutations_gp(Q, Qx_map, Y, x)
        self.model_permutations_gp([target_node], Qx_map, Y, x)

        self.model.setObjective(
            -CPT[target_node].flatten() @ Y[target_node] + x.sum(), GRB.MINIMIZE
        )
        end = time.time()
        self.model.optimize()
        self.model.update()
        total_time = end - start
        solver_time = self.model.Runtime
        print(
            f"Time taken to solve instance:{solver_time}\nTotal solving time:{total_time}\nOptimal path: {x.X}"
        )
        return total_time, solver_time
