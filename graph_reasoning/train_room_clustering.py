import numpy as np

def train_room_clustering(graphs):

    for graph in graphs:
        nodes_data = graph.nodes(data=True)
        print(nodes_data)