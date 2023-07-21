import numpy as np
import copy
import itertools
import random, math
import tqdm
from scipy.spatial.transform import Rotation as R
from graph_visualizer import visualize_nxgraph
from sklearn.neighbors import KDTree
from colorama import Fore, Back, Style

import sys
import os
graph_manager_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_manager","graph_manager")
sys.path.append(graph_manager_dir)
from GraphWrapper import GraphWrapper

class SyntheticDatasetGenerator():

    def __init__(self, settings):
        print(f"SyntheticDatasetGenerator:", Fore.GREEN + "Initializing" + Fore.WHITE)

        self.settings = settings

        self.define_norm_limits()
        self.create_dataset()

    def define_norm_limits(self):
        grid_dims = self.settings["base_graphs"]["grid_dims"]
        room_center_distances = self.settings["base_graphs"]["room_center_distances"]
        self.norm_limits = {}
        self.norm_limits["ws"] = {"min": np.array([-room_center_distances[0]/2,-room_center_distances[0]/2,-room_center_distances[0]/2, -room_center_distances[1]/2,-room_center_distances[1]/2,-room_center_distances[1]/2, -1, -1]),\
                        "max": np.array([room_center_distances[0]*grid_dims[0],room_center_distances[0]*grid_dims[0],room_center_distances[0]*grid_dims[0],room_center_distances[1]*grid_dims[1],room_center_distances[1]*grid_dims[1],room_center_distances[1]*grid_dims[1], 1,1])}


    def create_dataset(self):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Generating Syntetic Dataset" + Fore.WHITE)
        n_buildings = self.settings["base_graphs"]["n_buildings"]

        self.graphs = {"original":[],"noise":[],"views":[]}
        self.max_n_rooms = 0
        for n_building in tqdm.tqdm(range(n_buildings), colour="green"):
            base_matrix = self.generate_base_matrix()
            self.graphs["original"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False))
            self.graphs["noise"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= True))
            self.graphs["views"].append(self.generate_graph_from_base_matrix(base_matrix, add_noise= False, add_multiview=True))


    def generate_base_matrix(self):
        grid_dims = [np.random.randint(self.settings["base_graphs"]["grid_dims"][0][0], self.settings["base_graphs"]["grid_dims"][0][1] + 1),
                     np.random.randint(self.settings["base_graphs"]["grid_dims"][1][0], self.settings["base_graphs"]["grid_dims"][1][1] + 1)]
        max_room_entry_size = np.random.randint(self.settings["base_graphs"]["max_room_entry_size"][0], self.settings["base_graphs"]["max_room_entry_size"][1] + 1)

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

        self.max_n_rooms = max(self.max_n_rooms, room_n)
        return base_matrix


    def generate_graph_from_base_matrix(self, base_matrix, add_noise = False, add_multiview = False):
        graph = GraphWrapper()
        room_center_distances = self.settings["base_graphs"]["room_center_distances"]
        wall_thickness = self.settings["base_graphs"]["wall_thickness"]

        if add_noise:
            if self.settings["noise"]["global"]["active"]:
                noise_global_center = np.concatenate([np.array(self.settings["base_graphs"]["playground_size"]) * self.settings["noise"]["global"]["translation"] * (np.random.rand(2)- 0.5), [0]])
                noise_global_rotation_angle = (np.random.rand(1)*360*self.settings["noise"]["global"]["rotation"])[0]
                noise_global_rotation_angle=90
            else:
                noise_global_center = [0,0,0]
                noise_global_rotation_angle = 0

        ### Rooms
        for base_matrix_room_id in np.unique(base_matrix):
            occurrencies = np.argwhere(np.where(base_matrix == base_matrix_room_id, True, False))
            limits = [occurrencies[0],occurrencies[-1]]
            room_entry_size = [limits[1][0] - limits[0][0] + 1, limits[1][1] - limits[0][1] + 1]
            node_ID = len(graph.get_nodes_ids())
            room_center = np.array([room_center_distances[0]*(limits[0][0] + (room_entry_size[0]-1)/2), room_center_distances[1]*(limits[0][1]+(room_entry_size[1]-1)/2), 0])
            room_orientation_angle = 0.0
            room_area = [room_center_distances[0]*room_entry_size[0] - wall_thickness*2, room_center_distances[1]*room_entry_size[1] - wall_thickness*2, 0]
            if add_noise:
                if self.settings["noise"]["global"]["active"]:
                    room_orientation_angle += noise_global_rotation_angle

                if self.settings["noise"]["room"]["active"]:
                    center_noise = np.concatenate([np.random.rand(2)*room_center_distances*self.settings["noise"]["room"]["translation"], [0]])
                    room_orientation_angle += np.random.rand(1)[0]*360*self.settings["noise"]["room"]["rotation"]
                else:
                    center_noise = [0,0,0]
                
                room_center = R.from_euler("Z", noise_global_rotation_angle, degrees= True).apply(np.array(noise_global_center) + np.array(room_center) + center_noise)
                room_area = abs(R.from_euler("Z", room_orientation_angle, degrees= True).apply(room_area))
            geometric_info = room_center
            
            graph.add_nodes([(node_ID,{"type" : "room","center" : room_center, "x": [], "orientation_angle": room_orientation_angle, "area" : room_area, "Geometric_info" : geometric_info,\
                                            "viz_type" : "Point", "viz_data" : room_center[:2], "viz_feat" : 'bo'})])
        if add_multiview:
            num_multiviews = self.settings["multiview"]["number"]
            overlapping = self.settings["multiview"]["overlapping"]
            all_node_ids = graph.get_nodes_ids()
            masks = []
            for view_id in range(1, num_multiviews + 1):
                frontier = [int(len(all_node_ids)*(view_id-1)/ num_multiviews - np.random.randint(overlapping)),\
                    int(len(all_node_ids)* view_id / num_multiviews + np.random.randint(overlapping))]
                mask = [True if i in list(range(frontier[0], frontier[1])) else False for i in range(len(all_node_ids))]
                masks.append(mask)
            masks = np.array(masks)
            for i, node_id in enumerate(list(graph.get_nodes_ids())):
                graph.update_node_attrs(node_id, {"view" : np.squeeze(np.argwhere(masks[:, i]), axis= 1)+1})
        
        ### Wall surfaces
        room_nodes_data = copy.deepcopy(graph.get_attributes_of_all_nodes())
        canonic_normals = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
        for node_data in room_nodes_data:
            normals = copy.deepcopy(canonic_normals)
            if add_noise:
                if self.settings["noise"]["ws"]["active"]:
                    per_ws_noise_rot_angle = np.random.rand(4) * 360 * self.settings["noise"]["ws"]["rotation"]
                else:
                    per_ws_noise_rot_angle = [0,0,0,0]

                normals = np.array([list(R.from_euler("Z", node_data[1]["orientation_angle"] + per_ws_noise_rot_angle[j], degrees= True).apply(normals[j])) for j in range(4)])

            for i in range(4):
                node_ID = len(graph.get_nodes_ids())
                orthogonal_normal = R.from_euler("Z", 90, degrees= True).apply(copy.deepcopy(normals[i]))
                ws_normal = np.array([-1,-1, 0])*normals[i]
                ws_center = node_data[1]["center"] + vector_signed_projection(np.array(node_data[1]["area"])/2,np.array(normals[i]))
                # ws_center = node_data[1]["center"] + np.array(node_data[1]["area"])/2*np.array(normals[i])
                # ws_center = node_data[1]["center"] + np.array(normals[i])*np.array(node_data[1]["area"])/2
                
                ws_length = max(abs(np.array(orthogonal_normal)*np.array(node_data[1]["area"])))
                ws_limit_1 = ws_center + vector_signed_projection(np.array(node_data[1]["area"])/2,np.array(orthogonal_normal))
                ws_limit_2 = ws_center + vector_signed_projection(np.array(node_data[1]["area"])/2,np.array(-orthogonal_normal))
                x = np.concatenate([ws_center[:2], [ws_length], ws_normal[:2]]).astype(np.float32)
                # x_norm = (x-self.norm_limits["ws"]["min"])/(self.norm_limits["ws"]["max"]-self.norm_limits["ws"]["min"])
                x_norm = x
                self.len_ws_embedding = len(x)
                y = int(node_data[0])
                geometric_info = np.concatenate([ws_center, ws_normal])
                color_map = ["green", "orange", "red", "pink"]
                color_map = ["black", "black", "black", "black"]

                graph.add_nodes([(node_ID,{"type" : "ws","center" : ws_center, "x" : x_norm, "y" : y, "normal" : ws_normal, "Geometric_info" : geometric_info,\
                                           "viz_type" : "Line", "viz_data" : [ws_limit_1[:2],ws_limit_2[:2]], "viz_feat" : color_map[i],\
                                           "canonic_normal_index" : canonic_normals[i], "linewidth": 2.0, "limits": [ws_limit_1,ws_limit_2]})])
                graph.add_edges([(node_ID, node_data[0], {"type": "ws_belongs_room", "x": [], "viz_feat" : 'b', "linewidth":1.0, "alpha":0.5})])

                ### Fully connected version
                for prior_ws_i in range(i):
                    x = minimum_distance_two_wallsurfaces(graph.get_attributes_of_node(node_ID),graph.get_attributes_of_node(node_ID-(prior_ws_i+1)))
                    graph.add_edges([(node_ID, node_ID-(prior_ws_i+1), {"type": "ws_same_room", "x":x, "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # ### Only consecutive wall surfaces
                # if i > 0:
                #     graph.add_edges([(node_ID, node_ID - 1, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # if i == 3:
                #     graph.add_edges([(node_ID, node_ID - 3, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                # ### Only opposite wall surfaces
                # if i > 1:
                #     graph.add_edges([(node_ID, node_ID - 2, {"type": "ws_same_room", "viz_feat": "b", "linewidth":1.0, "alpha":0.5})])
                ###

                if add_multiview:
                    graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(node_data[0])["view"]})


        ### Walls

        explored_walls = []
        for i in range(base_matrix.shape[0]):
            for j in range(base_matrix.shape[1]):
                for ij_difference in [[1,0], [0,1]]:
                    ij_difference_3D = ij_difference + [0]
                    compared_ij = [i + ij_difference[0], j + ij_difference[1]]
                    current_room_id = base_matrix[i,j]
                    comparison = np.array(base_matrix.shape) > np.array(compared_ij)
                    if comparison.all() and current_room_id != base_matrix[compared_ij[0],compared_ij[1]]:
                        compared_room_id = base_matrix[compared_ij[0],compared_ij[1]]
                        if (current_room_id, compared_room_id) not in explored_walls:
                            explored_walls.append((current_room_id, compared_room_id))
                            current_room_neigh = graph.get_neighbourhood_graph(current_room_id-1).filter_graph_by_node_types(["ws"])
                            current_room_neigh_ws_id = list(current_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D}).get_nodes_ids())[0]
                            current_room_neigh_ws_center = current_room_neigh.get_attributes_of_node(current_room_neigh_ws_id)["center"]

                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            compared_room_neigh = graph.get_neighbourhood_graph(compared_room_id-1).filter_graph_by_node_types(["ws"])
                            ij_difference_3D_oppposite = list(-1*np.array(ij_difference_3D))
                            compared_room_neigh_ws_id = list(compared_room_neigh.filter_graph_by_node_attributes({"canonic_normal_index" : ij_difference_3D_oppposite}).get_nodes_ids())[0]
                            compared_room_neigh_ws_center = compared_room_neigh.get_attributes_of_node(compared_room_neigh_ws_id)["center"]

                            wall_center = list(np.array(current_room_neigh_ws_center) + (np.array(compared_room_neigh_ws_center) - np.array(current_room_neigh_ws_center))/2)
                            node_ID = len(graph.get_nodes_ids())
                            graph.add_nodes([(node_ID,{"type" : "wall","center" : wall_center,"viz_type" : "Point", "viz_data" : wall_center[:2], "viz_feat" : 'co'})])
                            graph.add_edges([(current_room_neigh_ws_id, node_ID, {"type": "ws_belongs_wall", "viz_feat": "c"}),(node_ID, compared_room_neigh_ws_id, {"type": "ws_belongs_wall",\
                                             "viz_feat": "c", "linewidth":1.0, "alpha":0.5})])
                            graph.add_edges([(current_room_neigh_ws_id, compared_room_neigh_ws_id, {"type": "ws_same_wall", "viz_feat": "c", "linewidth":1.0, "alpha":0.5})])
                            if add_multiview:
                                graph.update_node_attrs(node_ID, {"view" : graph.get_attributes_of_node(current_room_neigh_ws_id)["view"]})
        graph.to_undirected()
        return graph
    

    
    def get_filtered_datset(self, node_types, edge_types):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Filtering Dataset" + Fore.WHITE)
        nx_graphs = {}
        for key in self.graphs.keys():
            nx_graphs_key = []
            for base_graph in self.graphs[key]:
                filtered_graph = base_graph.filter_graph_by_node_types(node_types)
                filtered_graph.relabel_nodes() ### TODO What to do when Im dealing with different node types? Check tutorial
                filtered_graph = filtered_graph.filter_graph_by_edge_types(edge_types)
                nx_graphs_key.append(filtered_graph)
            nx_graphs[key] = nx_graphs_key

        return nx_graphs
    

    def extend_nxdataset(self, nxdataset):
        print(f"SyntheticDatasetGenerator: ", Fore.GREEN + "Extending Dataset" + Fore.WHITE)
        # hdataset = []
        new_nxdataset = []

        for i in tqdm.tqdm(range(len(nxdataset)), colour="green"):
            nxdata = nxdataset[i]
            base_graph = copy.deepcopy(nxdata)
            positive_gt_edge_ids = list(base_graph.get_edges_ids())
            if i == len(nxdataset)-1:
                settings = self.settings["postprocess"]["final"]
            else:
                settings = self.settings["postprocess"]["training"]

            ### Set positive label
            if settings["use_gt"]:
                for edge_id in list(base_graph.get_edges_ids()):
                    base_graph.update_edge_attrs(edge_id, {"label":1, "viz_feat" : 'green'})
            else:
                base_graph.remove_all_edges()

            ### Include K nearest neighbouors edges
            if settings["K_nearest"] > 0:
                node_ids = list(base_graph.get_nodes_ids())
                centers = np.array([attr[1]["center"] for attr in base_graph.get_attributes_of_all_nodes()])
                kdt = KDTree(centers, leaf_size=30, metric='euclidean')
                k = len(centers) if len(centers) <= settings["K_nearest"]+1 else settings["K_nearest"]+1
                query = kdt.query(centers, k=k, return_distance=False)
                query = np.array(list((map(lambda e: list(map(node_ids.__getitem__, e)), query))))
                base_nodes_ids = query[:, 0]
                all_target_nodes_ids = query[:, 1:]
                new_edges = []
                for i, base_node_id in enumerate(base_nodes_ids):
                    target_nodes_ids = all_target_nodes_ids[i]
                    for target_node_id in target_nodes_ids:
                        tuple_direct, tuple_inverse = (base_node_id, target_node_id), (target_node_id, base_node_id)
                        x = minimum_distance_two_wallsurfaces(base_graph.get_attributes_of_node(base_node_id),base_graph.get_attributes_of_node(target_node_id))
                        if tuple_direct in positive_gt_edge_ids or tuple_inverse in positive_gt_edge_ids:
                            if not settings["use_gt"]:
                                new_edges.append((base_node_id, target_node_id,{"type": "ws_same_room", "label": 1, "x":x, "viz_feat" : 'g', "linewidth":1.0, "alpha":0.5}))
                        else:
                            new_edges.append((base_node_id, target_node_id,{"type": "ws_same_room", "label": 0, "x":x, "viz_feat" : 'r', "linewidth":1.0, "alpha":0.5}))

                base_graph.unfreeze()
                base_graph.add_edges(new_edges)

            ### Include random edges
            if settings["K_random"] > 0:
                nodes_ids = list(base_graph.get_nodes_ids())
                for base_node_id in nodes_ids:
                    potential_nodes_ids = copy.deepcopy(nodes_ids)
                    potential_nodes_ids.remove(base_node_id)
                    random.shuffle(potential_nodes_ids)
                    random_nodes_ids = potential_nodes_ids[:settings["K_random"]]

                    new_edges = []
                    for target_node_id in random_nodes_ids:
                        tuple_direct = (base_node_id, target_node_id)
                        tuple_inverse = (tuple_direct[1], tuple_direct[0])
                        if tuple_direct not in list(base_graph.get_edges_ids()) and tuple_inverse not in list(base_graph.get_edges_ids()):
                            new_edges.append((tuple_direct[0], tuple_direct[1],{"type": "ws_same_room", "label": 0, "viz_feat" : 'blue', "linewidth":1.0, "alpha":0.5}))

                    base_graph.unfreeze()
                    base_graph.add_edges(new_edges)

            # hdata = from_networkxwrapper_2_heterodata(base_graph)
            # hdataset.append(hdata)
            new_nxdataset.append(base_graph)

        
        val_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["val"]-self.settings["training_split"]["test"]))
        test_start_index = int(len(nxdataset)*(1-self.settings["training_split"]["test"]))
        # hdataset_dict = {"train" : hdataset[:val_start_index], "val" : hdataset[val_start_index:test_start_index],"test" : hdataset[test_start_index:-1],"inference" : [hdataset[-1]]}
        extended_nxdatset = {"train" : new_nxdataset[:val_start_index], "val" : new_nxdataset[val_start_index:test_start_index],"test" : new_nxdataset[test_start_index:-1],"inference" : [new_nxdataset[-1]]}

        return extended_nxdatset


    def reintroduce_predicted_edges(self, unparented_base_graph, predictions, image_name = "name not provided"):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.add_edges(predictions)
        visualize_nxgraph(unparented_base_graph, image_name = image_name)

def vector_signed_projection(u,v):
    v_norm = np.sqrt(sum(v**2))
    proj_oj_u_on_v = abs(np.dot(u,v)/v_norm**2)*v
    return proj_oj_u_on_v

def minimum_distance_two_wallsurfaces(ws_1_def, ws_2_def):
    def minimum_distance_two_point_sets(set_1, set_2):
        min_distance = 99999
        for point_1 in set_1:
            for point_2 in set_2:
                dist = math.dist(point_1, point_2)
                if dist < min_distance:
                    min_distance = dist
        return np.array([min_distance])

    set_1 = np.concatenate([[np.array(ws_1_def["center"])], ws_1_def["limits"]])
    set_2 = np.concatenate([[np.array(ws_2_def["center"])], ws_2_def["limits"]])
    return minimum_distance_two_point_sets(set_1, set_2)
                

