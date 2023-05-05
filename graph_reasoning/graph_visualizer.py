import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_nxgraph(graph):
    nodes_data = graph.nodes(data=True)
    for node_data in nodes_data:
        if node_data[1]["viz_type"] == "Point":
            plt.plot(node_data[1]["viz_data"][0], node_data[1]["viz_data"][1], node_data[1]["viz_feat"])

        elif node_data[1]["viz_type"] == "Line":
            viz_data = np.array(node_data[1]["viz_data"])
            plt.plot(viz_data[:,0], viz_data[:,1], node_data[1]["viz_feat"])

    edges_data = graph.edges(data=True)
    for edge_data in edges_data:
        points = np.array([nodes_data[edge_data[0]]["center"], nodes_data[edge_data[1]]["center"]])
        plt.plot(points[:,0], points[:,1])

    plt.show()