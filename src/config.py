from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable


example_graph_nodes = {
    0: BbnNode(Variable(0, 0, ["state1", "state0"]), [0.5, 0.5]),
    1: BbnNode(Variable(1, 1, ["state1", "state0"]), [0.7, 0.3, 0.5, 0.5]),
    2: BbnNode(
        Variable(2, 2, ["state1", "state0"]), [0.9, 0.1, 0.4, 0.6, 0.5, 0.5, 0.1, 0.9]
    ),
    3: BbnNode(Variable(3, 3, ["state1", "state0"]), [0.6, 0.4, 0.2, 0.8]),
}


example_graph_edges = [
    Edge(example_graph_nodes[0], example_graph_nodes[1], EdgeType.DIRECTED),
    Edge(example_graph_nodes[1], example_graph_nodes[2], EdgeType.DIRECTED),
    Edge(example_graph_nodes[0], example_graph_nodes[2], EdgeType.DIRECTED),
    Edge(example_graph_nodes[2], example_graph_nodes[3], EdgeType.DIRECTED),
]
