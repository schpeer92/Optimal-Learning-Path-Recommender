import networkx as nx

from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType

from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

from pybbn.generator.bbngenerator import (
    generate_singly_bbn,
    generate_multi_bbn,
    convert_for_exact_inference,
    convert_for_drawing,
)


from config import example_graph_edges, example_graph_nodes


def nodes_to_jointree_n_graph(bbn):
    join_tree = InferenceController.apply(bbn)
    g, _ = bbn.to_nx_graph()
    return g, join_tree


def smallest_test_graph():
    """Returns networkx graph g and bbn join_tree with best path 1, 1"""

    a = BbnNode(Variable(0, 0, ["on", "off"]), [0.5, 0.5])
    b = BbnNode(Variable(1, 1, ["on", "off"]), [0.7, 0.3, 0.4, 0.6])
    c = BbnNode(Variable(2, 2, ["on", "off"]), [0.9, 0.1, 0.4, 0.6])

    # create the network structure
    bbn = (
        Bbn()
        .add_node(a)
        .add_node(b)
        .add_node(c)
        .add_edge(Edge(a, b, EdgeType.DIRECTED))
        .add_edge(Edge(b, c, EdgeType.DIRECTED))
    )

    # convert the BBN to a join tree
    return nodes_to_jointree_n_graph(bbn)


def eight_node_test_graph():
    ### create the nodes
    a = BbnNode(Variable(0, 0, ["on", "off"]), [0.5, 0.5])
    b = BbnNode(Variable(1, 1, ["on", "off"]), [0.5, 0.5, 0.4, 0.6])
    c = BbnNode(Variable(2, 2, ["on", "off"]), [0.7, 0.3, 0.2, 0.8])
    d = BbnNode(Variable(3, 3, ["on", "off"]), [0.9, 0.1, 0.5, 0.5])
    e = BbnNode(Variable(4, 4, ["on", "off"]), [0.3, 0.7, 0.6, 0.4])
    f = BbnNode(
        Variable(5, 5, ["on", "off"]), [0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.99, 0.01]
    )
    g = BbnNode(Variable(6, 6, ["on", "off"]), [0.8, 0.2, 0.1, 0.9])
    h = BbnNode(
        Variable(7, 7, ["on", "off"]), [0.05, 0.95, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05]
    )

    ### create the network structure
    bbn = (
        Bbn()
        .add_node(a)
        .add_node(b)
        .add_node(c)
        .add_node(d)
        .add_node(e)
        .add_node(f)
        .add_node(g)
        .add_node(h)
        .add_edge(Edge(a, b, EdgeType.DIRECTED))
        .add_edge(Edge(a, c, EdgeType.DIRECTED))
        .add_edge(Edge(b, d, EdgeType.DIRECTED))
        .add_edge(Edge(c, e, EdgeType.DIRECTED))
        .add_edge(Edge(d, f, EdgeType.DIRECTED))
        .add_edge(Edge(e, f, EdgeType.DIRECTED))
        .add_edge(Edge(c, g, EdgeType.DIRECTED))
        .add_edge(Edge(e, h, EdgeType.DIRECTED))
        .add_edge(Edge(g, h, EdgeType.DIRECTED))
    )

    # convert the BBN to a join tree
    return nodes_to_jointree_n_graph(bbn)


def example_graph_presentation():
    a = BbnNode(Variable(0, 0, ["state1", "state0"]), [0.5, 0.5])
    b = BbnNode(Variable(1, 1, ["state1", "state0"]), [0.7, 0.3, 0.5, 0.5])
    c = BbnNode(
        Variable(2, 2, ["state1", "state0"]), [0.9, 0.1, 0.4, 0.6, 0.5, 0.5, 0.1, 0.9]
    )
    d = BbnNode(Variable(3, 3, ["state1", "state0"]), [0.6, 0.4, 0.2, 0.8])

    bbn = (
        Bbn()
        .add_node(a)
        .add_node(b)
        .add_node(c)
        .add_node(d)
        .add_edge(Edge(a, b, EdgeType.DIRECTED))
        .add_edge(Edge(b, c, EdgeType.DIRECTED))
        .add_edge(Edge(a, c, EdgeType.DIRECTED))
        .add_edge(Edge(c, d, EdgeType.DIRECTED))
    )

    return nodes_to_jointree_n_graph(bbn)


def example_graph():

    bbn = Bbn()
    for node in example_graph_nodes.values():
        bbn.add_node(node)
    for edge in example_graph_edges:
        bbn.add_edge(edge)

    return bbn


def multiple_example_graph(n=2):
    g = example_graph()
    for _ in range(1, n):
        max_node_value = max(list(g.nodes))
        nodes_to_add = {
            max_node_value
            + 1: BbnNode(
                Variable(max_node_value + 1, max_node_value + 1, ["state1", "state0"]),
                [0.5, 0.5, 0.5, 0.5],
            ),
            max_node_value
            + 2: BbnNode(
                Variable(max_node_value + 2, max_node_value + 2, ["state1", "state0"]),
                [0.7, 0.3, 0.5, 0.5],
            ),
            max_node_value
            + 3: BbnNode(
                Variable(max_node_value + 3, max_node_value + 3, ["state1", "state0"]),
                [0.9, 0.1, 0.4, 0.6, 0.5, 0.5, 0.1, 0.9],
            ),
            max_node_value
            + 4: BbnNode(
                Variable(max_node_value + 4, max_node_value + 4, ["state1", "state0"]),
                [0.6, 0.4, 0.2, 0.8],
            ),
        }
        edges_to_add = [
            Edge(
                g.get_node(max_node_value),
                nodes_to_add[max_node_value + 1],
                EdgeType.DIRECTED,
            ),  # edge between old and new graph
            Edge(
                nodes_to_add[max_node_value + 1],
                nodes_to_add[max_node_value + 2],
                EdgeType.DIRECTED,
            ),
            Edge(
                nodes_to_add[max_node_value + 2],
                nodes_to_add[max_node_value + 3],
                EdgeType.DIRECTED,
            ),
            Edge(
                nodes_to_add[max_node_value + 1],
                nodes_to_add[max_node_value + 3],
                EdgeType.DIRECTED,
            ),
            Edge(
                nodes_to_add[max_node_value + 3],
                nodes_to_add[max_node_value + 4],
                EdgeType.DIRECTED,
            ),
        ]

        for node in nodes_to_add.values():
            g.add_node(node)
        for edge in edges_to_add:
            g.add_edge(edge)

    return nodes_to_jointree_n_graph(g)


def pybbn_generate_graph(func, number_of_nodes, max_iter):

    g, p = func(number_of_nodes, max_iter)
    bbn = convert_for_exact_inference(g, p)
    return nodes_to_jointree_n_graph(bbn)
