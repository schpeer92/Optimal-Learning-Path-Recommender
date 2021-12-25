import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt
from typing import Tuple
from gekko import GEKKO
from itertools import product
import numpy as np

# insert an observation evidence
def build_evidence(tree,node,value):
    if value==1:
        state = "on"
    else:
        state = "off"
    ev = EvidenceBuilder() \
    .with_node(tree.get_bbn_node(node)) \
    .with_evidence(state, value) \
    .build()
    tree.set_observation(ev)

def posterior_proba(tree,parent_node_states:list,node)-> float:
    for i,parent_state in enumerate(parent_node_states):
        if i!= node:
            build_evidence(tree,i,value.value)
    proba = join_tree.get_posteriors()[map_nx_to_bbn[node]] 
    
    tree.unobserve_all()
    return proba["on"]

def backwards_traversal(g:nx.DiGraph,source:int)->list:
    queue = [source]
    for node in queue:
        queue += list(g.predecessors(node))
    return queue

def forward_traversal(queue:list) -> list:
    x = [0 for i in range(len(queue))]


def create_CPT(g:nx.DiGraph,tree, node:int)->np.matrix:
    parents=list(g.predecessors(node))
    num_parents = len(parents)
    dimensions = tuple(2 for i in range(num_parents))
    cpt = np.array(dimensions)
    permutations = list(product(range(2),repeat = num_parents ))
    
    for permutation in permutations:
        posterior_proba(tree,permutation,node)
    return permutations


matrix_model = GEKKO()
m = matrix_model.Array(matrix_model.Const,8)#(4))
for i in range(8):
    m[i] = i*0.1


x = matrix_model.Array(matrix_model.Var,8, lb=0, ub=1, integer = True, value = 0)
z = matrix_model.Var(lb=0, ub=8, integer=True)

matrix_model.Equation(sum(x)==1)
for i in range(8):
    matrix_model.Equation(m[i] >= 0.5*x[i])
    matrix_model.Equation(sum(int(i) for i in np.binary_repr(i)) * x[i]<=z)


#TODO in need of optimization func
matrix_model.Minimize(-m@x +z)
matrix_model.options.SOLVER = 1
print(m)
matrix_model.solve(disp=True)

print(x )