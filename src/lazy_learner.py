from gekko import GEKKO
from itertools import product
import numpy as np
from pybbn.graph.jointree import EvidenceBuilder
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt
import time



import time
import functools


def timer(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure

class LazyLearner:

    def __init__(self, nx_graph,join_tree,
     ) -> None:
        self.G = nx_graph
        self.join_tree = join_tree

    @timer
    def plot_graph(self):
        my_pos = graphviz_layout(self.G, prog="dot" )
        nx.draw(self.G, pos = my_pos, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
        plt.show()

    @timer
    def build_evidence(self,node,value):
        state = "on" if value==1 else "off"
        ev = EvidenceBuilder() \
        .with_node(self.join_tree.get_bbn_node(node)) \
        .with_evidence(state, value) \
        .build()
        self.join_tree.set_observation(ev)

    
    @timer
    def posterior_proba(self,parent_node_states:tuple, permutation_to_node_map:tuple,node)-> float:
        for binary_value, parent_node in zip(parent_node_states, permutation_to_node_map):
            self.build_evidence(parent_node,binary_value)
        proba = self.join_tree.get_posteriors()[node] 
        
        self.join_tree.unobserve_all()
        return proba["on"]
    @timer
    def create_CPT(self,node:int) -> np.array:
        parents=tuple(self.G.predecessors(node))
        return (
            self.model.Array(self.model.Const,1,value = self.posterior_proba((), (), node))
            if parents == ()
            else self.create_CPT_with_parents(node)
        )
    @timer
    def parent_permutations(self, node)-> tuple:
        parents=tuple(self.G.predecessors(node))
        num_parents = len(parents)
        permutations = tuple(product(range(2),repeat = num_parents ))
        
        return permutations, parents
    @timer   
    def create_CPT_with_parents(self, node):
        permutations, parents = self.parent_permutations(node)
        num_parents = len(parents)
        dimensions = tuple(2 for i in range(num_parents))
        cpt = self.model.Array(self.model.Const, dimensions)
        
        for permutation in permutations:
            cpt[permutation] = self.posterior_proba(permutation,parents,node)
        return cpt
    @timer
    def backwards_traversal(self,target_node:int)->list:
        queue = list(self.G.predecessors(target_node))
        for node in queue:
            queue += list(self.G.predecessors(node))
        return queue
    @timer
    def create_CPT_for_all_nodes(self,Q, target_node):
        return {node: self.create_CPT(node) for node in Q+[target_node]}
    @timer
    def create_binary_variables(self,Q):
        return {node: self.model.Var(lb=0, ub=1, integer = True, value = 1) for node in Q}
    @timer
    def model_permutation_equations(self, cpt,x, realization, parents,target_node_value, threshold:float=0.5):
        if parents == ():    
            self.model.Equation(cpt[0]>= threshold*  x[target_node_value])
        elif target_node_value==1:
            is_taken = 1
            for instant_state,parent in zip(realization,parents):
                is_taken *= 1-x[parent] if instant_state == 0 else x[parent]
            self.model.Equation(cpt[realization] >= is_taken*threshold)
        else:
            is_taken = 1
            for instant_state,parent in zip(realization,parents):
                is_taken *= 1-x[parent] if instant_state == 0 else x[parent]
            self.model.Equation(cpt[realization] >= is_taken*threshold * x[target_node_value])
    @timer
    def model_permutations(self,node,x,cpt,threshold,target_node=False):
        permutations, parents = self.parent_permutations(node)
        for realization in permutations:
            if target_node:
                self.model_permutation_equations(cpt,x,realization,parents, 1,threshold)
            else:
                self.model_permutation_equations(cpt,x,realization,parents,node ,threshold)
    @timer
    def initialize_ip(self, target_node, threshold):
        start = time.time()
        self.model = GEKKO()
        threshold = self.model.Const(threshold)
        Q = self.backwards_traversal(target_node)
        CPT = self.create_CPT_for_all_nodes(Q, target_node)
        x = self.create_binary_variables(Q)
        
        #model IP for target node
        for node in Q:
            self.model_permutations(node,x,CPT[node],threshold, target_node=False)
        
        #model IP for all pare nodes (if taken property)
        self.model_permutations(target_node,x, CPT[target_node], threshold, target_node=True)
        
        #only optimizing parent path - other might yield problems
        _,target_parents = self.parent_permutations(target_node)
        y = self.model.Array(self.model.Var, 2**len(target_parents))
        z = self.model.Var()

 

        target_parents_variables = [parent for parent in Q if parent in target_parents]
        self.model.Equation(z == sum(2**i * x[xi] for i, xi in enumerate(target_parents_variables)))

        for i,yi in enumerate(y):
            self.model.Equation(yi == 0**((i-z)**2))

        self.model.Minimize(  self.model.sum([x[i] for i in x]) - CPT[target_node].flatten()@y)
        self.model.options.SOLVER = 1

        self.model.solve(disp=False)
        end = time.time()
        total_time = end-start
        solver_time = self.model.options.SOLVETIME
        print(f"Time taken to solve instance:{solver_time}\nTotal solving time:{total_time}\nOptimal path: {x}")
        return total_time,solver_time
        
        
