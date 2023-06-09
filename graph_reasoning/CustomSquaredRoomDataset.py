import numpy as np
import networkx as nx
from graph_visualizer import visualize_nxgraph
from sklearn.neighbors import KDTree
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import HeteroData


import sys
import os
graph_manager_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_manager","graph_manager")
sys.path.append(graph_manager_dir)
from GraphWrapper import GraphWrapper

class CustomSquaredRoomNetworkxGraphs():

    def __init__(self, settings):
        print(f"CustomSquaredRoomNetworkxGraphs: Initializing")

        self.synthetic_dataset_settings = settings["synthetic_datset"]
        grid_dims = self.synthetic_dataset_settings["grid_dims"]
        room_center_distances = self.synthetic_dataset_settings["room_center_distances"]
        wall_thickness = self.synthetic_dataset_settings["wall_thickness"]
        max_room_entry_size = self.synthetic_dataset_settings["max_room_entry_size"]
        n_buildings = self.synthetic_dataset_settings["n_buildings"]
        
        self.define_norm_limits(grid_dims, room_center_distances, wall_thickness, max_room_entry_size, n_buildings)
        self.generate_base_graphs(grid_dims, room_center_distances, wall_thickness, max_room_entry_size, n_buildings)

    def define_norm_limits(self, grid_dims, room_center_distances, wall_thickness, max_room_entry_size, n_buildings):
        self.norm_limits = {}
        self.norm_limits["ws"] = {"min": np.array([-room_center_distances[0]/2, -room_center_distances[1]/2, -1, -1, 0]),\
                        "max": np.array([room_center_distances[0]*grid_dims[0],room_center_distances[1]*grid_dims[1], 1,1, room_center_distances[1]*grid_dims[1]])}

    def generate_base_graphs(self, grid_dims, room_center_distances, wall_thickness, max_room_entry_size, n_buildings):
        print(f"CustomSquaredRoomNetworkxGraphs: Generating base graphs")
        base_graphs = []
        self.max_n_rooms = 0
        for floor_n in range(n_buildings):
            graph = GraphWrapper()

            ### Base matrix
            base_matrix = np.zeros(grid_dims)
            room_n = 0
            for i in range(base_matrix.shape[0]):
                for j in range(base_matrix.shape[1]):
                    if base_matrix[i,j] == 0.:
                        room_n += 1
                        aux_col = np.where(base_matrix[i:,j] != 0)[0]
                        aux_row = np.where(base_matrix[i,j:] != 0)[0]
                        if len(aux_col) != 0:
                            remaining_x = aux_col[0]
                        else:
                            remaining_x = len(base_matrix[i:,j])
                        if len(aux_row) != 0:
                            remaining_y = aux_row[0]
                        else:
                            remaining_y = len(base_matrix[i,j:])
                        remaining = [remaining_x, remaining_y]
                        room_entry_size = [min(remaining[0], np.random.randint(low=1, high=max_room_entry_size+1, size=(1))[0]),\
                                        min(remaining[1], np.random.randint(low=1, high=max_room_entry_size+1, size=(1))[0])]
                        for ii in range(room_entry_size[0]):
                            for jj in range(room_entry_size[1]):
                                base_matrix[i+ii, j+jj] = room_n

                        node_ID = len(graph.get_nodes_ids())
                        room_center = [room_center_distances[0]*(i + ii/2), room_center_distances[1]*(j+jj/2)]
                        room_area = [room_center_distances[0]*room_entry_size[0] - wall_thickness*2, room_center_distances[1]*room_entry_size[1] - wall_thickness*2]
                        graph.add_nodes([(node_ID,{"type" : "room","center" : room_center, "room_area" : room_area,\
                                                        "viz_type" : "Point", "viz_data" : room_center, "viz_feat" : 'bx'})])
                        

            total_rooms_n = room_n
            self.max_n_rooms = max(self.max_n_rooms, total_rooms_n)

            ### Wall surfaces
            nodes_data = copy.deepcopy(graph.get_attributes_of_all_nodes())
            for node_data in nodes_data:
                normals = [[1,0],[-1,0],[0,1],[0,-1]]
                for i in range(4):
                    node_ID = len(graph.get_nodes_ids())
                    orthogonal_normal = np.rot90([normals[i]]).reshape(2)
                    ws_normal = np.array([-1,-1])*normals[i]
                    ws_center = node_data[1]["center"] + np.array(normals[i])*np.array(node_data[1]["room_area"])/2
                    ws_length = max(abs(np.array(orthogonal_normal)*np.array(node_data[1]["room_area"])))
                    ws_limit_1 = ws_center + np.array(orthogonal_normal)*np.array(node_data[1]["room_area"])/2
                    ws_limit_2 = ws_center + np.array(-orthogonal_normal)*np.array(node_data[1]["room_area"])/2
                    x = np.concatenate([ws_center, ws_normal, [ws_length]]).astype(np.float32) # TODO Not sure of this
                    # print(f"x {x}")
                    x_norm = (x-self.norm_limits["ws"]["min"])/(self.norm_limits["ws"]["max"]-self.norm_limits["ws"]["min"])
                    # print(f"x_norm {x_norm}")
                    self.len_ws_embedding = len(x)
                    y = int(node_data[0])
                    graph.add_nodes([(node_ID,{"type" : "ws","center" : ws_center, "x" : x_norm, "y" : y, "normal" : ws_normal,\
                                                 "viz_type" : "Line", "viz_data" : [ws_limit_1,ws_limit_2], "viz_feat" : 'k'})])
                    graph.add_edges([(node_ID, node_data[0], {"x": []})])
                    for prior_ws_i in range(i):
                        graph.add_edges([(node_ID, node_ID-(prior_ws_i+1), {"type": "ws_same_room", "viz_feat": ""})])

            ## Walls

            # explored_walls = []
            # for i in range(base_matrix.shape[0]):
            #     for j in range(base_matrix.shape[1]):
            #         for ij_difference in [[1,0], [0,1]]:
            #             compared_ij = [i + ij_difference[0], j + ij_difference[1]]
            #             current_room_id = base_matrix[i,j]
            #             comparison = np.array(base_matrix.shape) > np.array(compared_ij)
            #             if comparison.all() and current_room_id != base_matrix[compared_ij[0],compared_ij[1]]:
            #                 compared_room_id = base_matrix[compared_ij[0],compared_ij[1]]
            #                 if (current_room_id, compared_room_id) not in explored_walls:
            #                     explored_walls.append((current_room_id, compared_room_id))
            #                     current_room_neigh_ws_ids = graph.get_neighbourhood_graph(current_room_id-1).filter_graph_by_node_types(["ws"]).get_nodes_ids()
            #                     current_room_expected_normal = list(np.array([-1,-1])*np.array(ij_difference))
            #                     for id, neigh_id in enumerate(current_room_neigh_ws_ids):
            #                         if (graph.get_attributes_of_node(neigh_id)["normal"] == current_room_expected_normal).all():
            #                             current_room_neigh_ws_id = neigh_id
            #                             current_room_neigh_ws_center = graph.get_attributes_of_node(neigh_id)["center"]

            #                     compared_room_neigh_ws_ids = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"]).get_nodes_ids()
            #                     compared_room_expected_normal = ij_difference
            #                     for id, neigh_id in enumerate(compared_room_neigh_ws_ids):
            #                         if (graph.get_attributes_of_node(neigh_id)["normal"] == compared_room_expected_normal).all():
            #                             compared_room_neigh_ws_id = neigh_id
            #                             compared_room_neigh_ws_center = graph.get_attributes_of_node(neigh_id)["center"]

            #                     wall_center = list(np.array(current_room_neigh_ws_center) + (np.array(compared_room_neigh_ws_center) - np.array(current_room_neigh_ws_center))/2)
            #                     node_ID = len(graph.get_nodes_ids())
            #                     # graph.add_nodes([(node_ID,{"type" : "wall","center" : wall_center,"viz_type" : "Point", "viz_data" : wall_center, "viz_feat" : 'gx'})])
            #                     # graph.add_edges([(current_room_neigh_ws_id, node_ID, {}),(node_ID, compared_room_neigh_ws_id, {})])
            #                     graph.add_edges([(current_room_neigh_ws_id, compared_room_neigh_ws_id, {"type": "ws_same_wall", "viz_feat": "p"})])
                                   

            base_graphs.append(graph)

            # visualize_nxgraph(graph, image_name = "Synthetic data creation: example")

        self.base_graphs = base_graphs


    def get_ws2room_clustering_datalodaer(self):
        val_ratio = self.synthetic_dataset_settings["val_ratio"]
        test_ratio = self.synthetic_dataset_settings["test_ratio"]
        nx_graphs = []
        for base_graph in self.base_graphs:
            room_graph = base_graph.filter_graph_by_node_types(["ws"])
            room_graph.relabel_nodes()
            # visualize_nxgraph(room_graph)
            nx_graphs.append(room_graph)

        return nx_graphs
    

    def get_ws2room_clustering_single_base_knn_graph(self, visualize = False):
        gt_base_graph = copy.deepcopy(self.base_graphs[np.random.randint(len(self.base_graphs))].filter_graph_by_node_types(["ws"]))
        print(f"flag base graph edge_index {min(gt_base_graph.get_nodes_ids())}")
        node_label_mapping = gt_base_graph.relabel_nodes()
        print(f"flag 2 base graph edge_index {min(gt_base_graph.get_nodes_ids())}")
        visualize_nxgraph(gt_base_graph, image_name = "Inference: base synthetic graph") if visualize else None
        ground_truth = list(gt_base_graph.filter_graph_by_node_types(["ws"]).get_edges_ids())
        base_graph = copy.deepcopy(gt_base_graph)
        
        base_graph.remove_all_edges()
        visualize_nxgraph(base_graph, image_name = "Inference: base synthetic graph only WSs") if visualize else None
        
        node_indexes = list(base_graph.get_nodes_ids())
        centers = np.array([attr[1]["center"] for attr in base_graph.get_attributes_of_all_nodes()])
        kdt = KDTree(centers, leaf_size=30, metric='euclidean')
        query = kdt.query(centers, k=self.synthetic_dataset_settings["K_nn"], return_distance=False)[:, 1:]
        new_edges = []
        for base_node, target_nodes in enumerate(query):
            for target_node in target_nodes:
                new_edges.append((node_indexes[base_node], node_indexes[target_node],{"type": "ws_same_room"}))
        base_graph.add_edges(new_edges)
        visualize_nxgraph(base_graph, image_name = "Inference: base synthetic graph auxiliar raw edges") if visualize else None
        # node_label_mapping = base_graph.relabel_nodes()
        unparented_base_graph = copy.deepcopy(base_graph)
        unparented_base_graph.remove_all_edges()
        # print(f"num nodes in mapping {len(node_label_mapping.keys())}")
        ground_truth_directed = []
        for gt in ground_truth:
            ground_truth_directed.append((gt[0], gt[1]))
            ground_truth_directed.append((gt[1], gt[0]))
        gt_edges = [(gt[0], gt[1], {"type" : "ws_same_room"}) for gt in ground_truth_directed]

        return gt_base_graph, unparented_base_graph, base_graph, node_label_mapping, ground_truth_directed, gt_edges
    
    def reintroduce_predicted_edges(self, unparented_base_graph, predictions, image_name = "name not provided"):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.add_edges(predictions)
        visualize_nxgraph(unparented_base_graph, image_name = image_name)