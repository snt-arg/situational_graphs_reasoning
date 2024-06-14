from torch_geometric.data import HeteroData

import os, sys
import numpy as np
from torch import Tensor
import torch
import torch_geometric.transforms as T

from situational_graphs_wrapper.GraphWrapper import GraphWrapper
from situational_graphs_datasets.graph_visualizer import visualize_nxgraph


def from_networkxwrapper_2_heterodata(networkx_graph):
    hdata = HeteroData()
    # visualize_nxgraph(networkx_graph, image_name= "pre-hdata")
    # networkx_graph.draw(fig_name= "pre-hdata", options = None, show = True)
    node_types = networkx_graph.get_all_node_types()

    for node_type in node_types:
        subgraph = networkx_graph.filter_graph_by_node_types([node_type])
        hdata[node_type].node_id =  Tensor(np.array(list(subgraph.get_nodes_ids())).astype(int)).to(torch.int64).contiguous()
        hdata[node_type].x = torch.from_numpy(np.array([attr[1]["x"] for attr in subgraph.get_attributes_of_all_nodes()])).to(torch.float).contiguous() 

    edge_types = networkx_graph.get_all_edge_types()
    for edge_type in edge_types:
        subgraph = networkx_graph.filter_graph_by_edge_types([edge_type])
        edges_ids = np.array(subgraph.get_edges_ids()).transpose().astype(int)
        n1_type, n2_type = subgraph.get_attributes_of_node(edges_ids[0][0])["type"], subgraph.get_attributes_of_node(edges_ids[0][1])["type"]
        hdata[n1_type, edge_type, n2_type].edge_index = Tensor(edges_ids).to(torch.int64).contiguous()
        hdata[n1_type, edge_type, n2_type].x = torch.from_numpy(np.array([attr[2]["x"] for attr in subgraph.get_attributes_of_all_edges()])).to(torch.float).contiguous() 
        
    for edge_type in edge_types:
        edges_attrs = list(subgraph.get_attributes_of_all_edges())
        if "label" in edges_attrs[0][2].keys():
            hdata[n1_type, edge_type, n2_type].edge_label = torch.from_numpy(np.array([attr[2]["label"] for attr in edges_attrs])).to(torch.long).contiguous()
            # print(f"flag hdata[n1_type, edge_type, n2_type].edge_label {hdata[n1_type, edge_type, n2_type].edge_label} ")
            # hdata[n1_type, edge_type, n2_type].edge_label_index = Tensor(edges_ids).to(torch.int64).contiguous()

    # print(f"flag 1 hdata {hdata}")
    # hdata = T.ToUndirected(merge=False)(hdata)
    # for edge_type in edge_types:
    #     del  hdata[n1_type, "rev_" + edge_type, n2_type].edge_label
    # print(f"flag 2 hdata {hdata}")
    # sdfg
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
