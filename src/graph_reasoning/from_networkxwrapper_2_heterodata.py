from torch_geometric.data import HeteroData

import os, sys, copy
import numpy as np
import time
from torch import Tensor
import torch
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
sys.path.append(graph_wrapper_dir)
from graph_wrapper.GraphWrapper import GraphWrapper
graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph


def from_networkxwrapper_2_heterodata(networkx_graph):
    hdata = HeteroData()
    # visualize_nxgraph(networkx_graph, image_name= "pre-hdata", visualize_alone = True)
    node_types = networkx_graph.get_all_node_types()

    for node_type in node_types:
        subgraph = networkx_graph.filter_graph_by_node_types([node_type])
        hdata[node_type].node_id = Tensor(np.array(list(subgraph.get_nodes_ids())).astype(int)).to(torch.int64).contiguous()
        # print(f"dbg subgraph.get_attributes_of_all_nodes()[0]['x'] {subgraph.get_attributes_of_all_nodes()[0]['x']}")
        hdata[node_type].x = torch.from_numpy(np.array([attr[1]["x"] for attr in subgraph.get_attributes_of_all_nodes()])).to(torch.float).contiguous() 

    edge_types = sorted(list(networkx_graph.get_all_edge_types()))
    for edge_type in edge_types:
        subgraph = networkx_graph.filter_graph_by_edge_types([edge_type])
        edges_ids = np.array(subgraph.get_edges_ids()).transpose().astype(int)
        # print(f"dbg subgraph.get_attributes_of_all_edges()[0][2][x] {list(subgraph.get_attributes_of_all_edges())[0][2]['x']}")
        for i in range(len(edges_ids[0])):
            n1_type, n2_type = subgraph.get_attributes_of_node(edges_ids[0][i])["type"], subgraph.get_attributes_of_node(edges_ids[1][i])["type"]
            hdata[n1_type, edge_type, n2_type].edge_index = Tensor(edges_ids).to(torch.int64).contiguous()
            hdata[n1_type, edge_type, n2_type].x = torch.from_numpy(np.array([attr[2]["x"] for attr in subgraph.get_attributes_of_all_edges()])).to(torch.float).contiguous() 
    
    for edge_type in edge_types:
        edges_attrs = list(subgraph.get_attributes_of_all_edges())
        if "label" in edges_attrs[0][2].keys():
            # print(f"dbg attr[2][label] {edges_attrs[0][2]['label']}")
            hdata[n1_type, edge_type, n2_type].edge_label = torch.from_numpy(np.array([attr[2]["label"] for attr in edges_attrs])).to(torch.long).contiguous()
            # print(f"flag hdata[n1_type, edge_type, n2_type].edge_label {hdata[n1_type, edge_type, n2_type].edge_label} ")
            # hdata[n1_type, edge_type, n2_type].edge_label_index = Tensor(edges_ids).to(torch.int64).contiguous()

    if not networkx_graph.is_directed():
        hdata = T.ToUndirected(merge=True)(hdata)

    node_key = list(hdata.edge_index_dict.keys())[0][0]
    edge_key = list(hdata.edge_index_dict.keys())[0][1]
    edge_index = copy.copy(hdata.edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
    edge_index_tuples = list(zip(edge_index[0], edge_index[1]))
    edge_label_index = copy.copy(hdata.edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
    edge_label_index_tuples_compressed = np.array(list({tuple(sorted((edge_label_index[0, i], edge_label_index[1, i]))) for i in range(edge_label_index.shape[1])}))
    edge_label_index_tuples_compressed_inversed = edge_label_index_tuples_compressed[:, ::-1]
    src, dst = edge_label_index_tuples_compressed[:,0], edge_label_index_tuples_compressed[:,1]
    edge_index_to_edge_label_index = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed]
    edge_index_to_edge_label_index_inversed = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed_inversed]
    hdata.edge_label_dict = {"src":src, "dst":dst, "edge_index_to_edge_label_index":edge_index_to_edge_label_index, "edge_index_to_edge_label_index_inversed":edge_index_to_edge_label_index_inversed, "edge_label_index_tuples_compressed":edge_label_index_tuples_compressed}

    return hdata

def from_heterodata_2_networkxwrapper(hdata):

    graph = GraphWrapper()
    for node_type in hdata.node_types:
        hdata_node_type = hdata[node_type]
        tuples = []
        for i in range(len(hdata_node_type["node_id"])):
            tuples.append((hdata_node_type["node_id"][i], {"type": node_type}))
        graph.add_nodes(tuples)

    for edge_type in hdata.edge_types:
        hdata_edge_type = hdata[edge_type[0], edge_type[1], edge_type[2]]
        tuples = []
        for i in range(len(hdata_edge_type.edge_index[0])):
            tuples.append((edge_type[0], edge_type[2], {"type": edge_type[1]}))
        graph.add_edges(tuples)
    # graph.draw(fig_name= "post-hdata", options = None, show = True)
    # visualize_nxgraph(graph, image_name= "post-hdata")
