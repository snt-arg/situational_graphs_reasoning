# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import time, os, json, shutil, sys, torch
import copy
import numpy as np
import ament_index_python
import argparse
import ast
from rclpy.node import Node
# from tf2_ros.transform_listener import TransformListener
# from tf2_ros.buffer import Buffer
# from tf2_ros.buffer_interface import BufferInterface
# import tf2_geometry_msgs 
# from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg
from geometry_msgs.msg import Pose as PoseMsg
# from geometry_msgs.msg import Vector3 as Vector3Msg
# from geometry_msgs.msg import PointStamped as PointStampedMsg
from geometry_msgs.msg import Point as PointMsg
# from geometry_msgs.msg import Transform as TransformMsg
# from geometry_msgs.msg import TransformStamped as TransformStampedMsg
# from std_msgs.msg import ColorRGBA as ColorRGBSMsg
# from std_msgs.msg import Header as HeaderMsg
# from builtin_interfaces.msg import Duration as DurationMsg
# from rclpy.parameter import Parameter
# from rclpy.parameter import ParameterType
# from ament_index_python.packages import get_package_share_directory
from shapely.geometry import Polygon

from situational_graphs_msgs.msg import PlanesData as PlanesDataMsg
from situational_graphs_msgs.msg import RoomsData as RoomsDataMsg
from situational_graphs_msgs.msg import RoomData as RoomDataMsg
from situational_graphs_msgs.msg import WallsData as WallsDataMsg
from situational_graphs_msgs.msg import WallData as WallDataMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg

from graph_reasoning.GNNWrapper import GNNWrapper
from graph_reasoning.EvolvingSetsTracker import EvolvingSetsTracker
from graph_reasoning.config import get_config as reasoning_get_config
from graph_reasoning.pths import get_pth as reasoning_get_pth
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.config import get_config as datasets_get_config
from graph_matching.utils import segments_distance, segment_intersection, plane_6_params_to_4_params
from graph_factor_nn.FactorNNBridge import FactorNNBridge

import math

graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph

class GraphReasoningNode(Node):
    def __init__(self, args):
        super().__init__('graph_reasoning')

        args = self.parse_arguments(args)
        
        # args = ["room", "wall"]
        self.get_logger().info(f"dbg args {args}")
        self.find_rooms, self.find_walls, self.find_floors, self.find_RoomWall = False, False, False, False
        
        self.use_gnn_factors = args.use_gnn_factors
        if self.use_gnn_factors:
            self.factor_nn = FactorNNBridge(["room", "wall", "floor"])

        self.concept_set_trackers = {}
        if "room" in args.generated_entities:
            self.find_rooms = True
            self.concept_set_trackers["room"] = EvolvingSetsTracker()
        if "wall" in args.generated_entities:
            self.find_walls = True
            self.concept_set_trackers["wall"] = EvolvingSetsTracker()
        if "floor" in args.generated_entities:
            self.find_floors = True
        if "RoomWall" in args.generated_entities:
            self.find_RoomWall = True
            self.concept_set_trackers["room"] = EvolvingSetsTracker()
            self.concept_set_trackers["wall"] = EvolvingSetsTracker()

        self.generation_plots_path = args.log_path + "/generation_plots"
        os.makedirs(self.generation_plots_path)
        self.generation_i = 0

        # self.graph_reasoning_rooms_settings = reasoning_get_config("same_room_best")
        # self.graph_reasoning_walls_settings = reasoning_get_config("same_wall_best")
        # self.graph_reasoning_floors_settings = reasoning_get_config("same_floor_training")
        self.graph_reasoning_RoomWall_settings = reasoning_get_config("same_RoomWall_best")
        self.reasoning_package_path = ament_index_python.get_package_share_directory("graph_reasoning")

        dataset_settings = datasets_get_config("graph_reasoning")

        dataset_settings["training_split"]["val"] = 0.0
        dataset_settings["training_split"]["test"] = 0.0
        
        self.dataset_settings = dataset_settings
        self.elapsed_times = []
        self.prepare_report_folder()
        
        self.gnns = {}
        if self.find_rooms:
            self.gnns.update({"room": GNNWrapper(self.graph_reasoning_rooms_settings, self.report_path, self.get_logger())})
            self.gnns["room"].define_GCN()
            # self.gnns["room"].pth_path = os.path.join(self.reasoning_package_path, "pths/model_rooms.pth")
            self.gnns["room"].pth_path = reasoning_get_pth("model_rooms_best")
            self.gnns["room"].load_model()
            self.gnns["room"].save_model(os.path.join(self.report_path,"model_rooms_best.pth"))
        if self.find_walls:
            self.gnns.update({"wall": GNNWrapper(self.graph_reasoning_walls_settings, self.report_path, self.get_logger())})
            self.gnns["wall"].define_GCN()
            self.gnns["wall"].pth_path = reasoning_get_pth("model_walls_best")
            self.gnns["wall"].load_model() 
            self.gnns["wall"].save_model(os.path.join(self.report_path,"model_walls_best.pth")) 
        if self.find_floors:
            self.gnns.update({"floor": GNNWrapper(self.graph_reasoning_floors_settings, self.report_path, self.get_logger())})
            self.gnns["floor"].define_GCN()
            # self.gnns["floor"].pth_path = os.path.join(self.reasoning_package_path, "pths/model_floors.pth")
            # self.gnns["floor"].load_model() 
            # self.gnns["floor"].save_model(os.path.join(self.report_path,"model_floor.pth")) 
        if self.find_RoomWall:
            self.gnns.update({"RoomWall": GNNWrapper(self.graph_reasoning_RoomWall_settings, self.report_path, self.get_logger())})
            self.gnns["RoomWall"].define_GCN()
            self.gnns["RoomWall"].pth_path = reasoning_get_pth("model_RoomWall_best")
            self.gnns["RoomWall"].load_model() 
            self.gnns["RoomWall"].save_model(os.path.join(self.report_path,"model_RoomWall_best.pth"))

        self.synthetic_dataset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger(), self.report_path)
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")
        self.node_start_time = time.perf_counter()
        self.first_room_detected = False
        self.tmp_room_history = []
        self.generation_times_history = []
  

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","sgraphs", "inference")
        self.get_logger().info(f"{self.report_path}")
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path)
        else:
            for filename in os.listdir(self.report_path):
                file_path = os.path.join(self.report_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        # combined_settings = {"dataset": self.dataset_settings, "graph_reasoning_rooms": self.graph_reasoning_rooms_settings,\
        #                      "graph_reasoning_walls": self.graph_reasoning_walls_settings, "graph_reasoning_floors": self.graph_reasoning_floors_settings,\
        #                      "graph_reasoning_RoomWall": self.graph_reasoning_RoomWall_settings}
        combined_settings = {"dataset": self.dataset_settings, "graph_reasoning_RoomWall": self.graph_reasoning_RoomWall_settings}
        with open(os.path.join(self.report_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)

    def set_interface(self):
        if self.find_walls:
            self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_all_planes_callback, 10)
            self.wall_subgraph_publisher = self.create_publisher(WallsDataMsg, '/wall_segmentation/wall_data', 10)
        if self.find_rooms:
            self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/map_planes', self.s_graph_last_planes_callback, 10)
            self.room_subgraph_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)
        if self.find_floors:
            self.s_graph_subscription = self.create_subscription(MarkerArrayMsg,'/s_graphs/markers', self.s_graph_room_marker_callback, 10)
            self.floor_subgraph_publisher = self.create_publisher(RoomDataMsg, '/floor_plan/floor_data', 10)
        if self.find_RoomWall:
            self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_all_planes_callback, 10)
            self.wall_subgraph_publisher = self.create_publisher(WallsDataMsg, '/wall_segmentation/wall_data', 10)
            self.room_subgraph_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)

    def s_graph_all_planes_callback(self, msg):
        # start_time = time.time()
        self.infer_from_planes(msg)
        # end_time = time.time()
        # self.generation_times_history.append(end_time - start_time)
        # averaged_generation_times_history = sum(self.generation_times_history)/len(self.generation_times_history)
        # print(f"dbg averaged_generation_times_history {averaged_generation_times_history}")

    def s_graph_last_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in LAST planes topic")
        self.infer_from_planes("room", msg)

    def s_graph_room_marker_callback(self, msg):
        # self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in LAST planes topic")
        self.infer_from_rooms("floor", msg)

    def infer_from_planes(self, msg):
        if len(msg.x_planes) == 0 or len(msg.y_planes) == 0:
            return
        
        target_concept = "RoomWall"
        
        graph = GraphWrapper()
        graph.to_directed()
        initial_filtered_planes_graph = GraphWrapper()
        initial_filtered_planes_graph.to_directed()
        planes_msgs = msg.x_planes + msg.y_planes
        # planes_msgs = self.dbg_fake_plane_msgs() ### DBG
        planes_dicts = []
        for i, plane_msg in enumerate(planes_msgs):
            if len(plane_msg.plane_points) != 0:
                plane_dict = {"id": plane_msg.id, "normal" : np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])}
                plane_dict["xy_type"] = "x" if i<len(msg.x_planes) else "y"
                plane_dict["msg"] = plane_msg
                plane_dict["center"], plane_dict["segment"], plane_dict["length"] = self.characterize_ws(plane_msg.plane_points)
                planes_dicts.append(plane_dict)

        ### Degug
        # initial_planes_graph = GraphWrapper()
        # initial_planes_graph.to_directed()
        # for plane_dict in planes_dicts:
        #     initial_planes_graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "label": 1, "normal" : plane_dict["normal"],\
        #                             "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
        #                             "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
        # fig = visualize_nxgraph(initial_planes_graph, image_name = f"filtered input from sgraphs", include_node_ids= True, visualize_alone=False)
        # fig.savefig(self.generation_plots_path + f"/input_from_sgraph_{self.generation_i}.png")

        ### Debug End

        filtered_planes_dicts = self.filter_overlapped_ws(planes_dicts)
        filtered_planes_dicts_dict = {plane_dict["id"]: plane_dict for plane_dict in filtered_planes_dicts}
        # self.get_logger().info(f"dbg filtered_planes ids {filtered_planes_dicts_dict.keys()}")
        for plane_dict in filtered_planes_dicts:
            initial_filtered_planes_graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "label": 1, "normal" : plane_dict["normal"],\
                                    "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
                                    "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
        # fig = visualize_nxgraph(initial_filtered_planes_graph, image_name = f"filtered input from sgraphs", include_node_ids= True, visualize_alone=False)
        # fig.savefig(self.report_path + "/input_from_sgraph.png")

        splitted_planes_dicts = self.split_ws(filtered_planes_dicts)
        splitting_mapping = {}
        for plane_dict in splitted_planes_dicts:
            def add_ws_node_features(feature_keys, feats):
                if feature_keys[0] == "centroid":
                    feats = np.concatenate([feats, plane_dict["center"][:2]]).astype(np.float32)
                elif feature_keys[0] == "length":
                    feats = np.concatenate([feats, [plane_dict["length"]]]).astype(np.float32)   #, [np.log(ws_length)]]).astype(np.float32)
                elif feature_keys[0] == "normals":
                    feats = np.concatenate([feats, plane_dict["normal"][:2]]).astype(np.float32)
                if len(feature_keys) > 1:
                    feats = add_ws_node_features(feature_keys[1:], feats)
                return feats
            x = add_ws_node_features(self.dataset_settings["initial_features"]["nodes"]["ws"], [])

            graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "x" : x, "label": 1, "normal" : plane_dict["normal"],\
                                           "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
                                           "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
            splitting_mapping[plane_dict["id"]] = plane_dict["old_id"]

        # Inference
        graph.to_directed()
        extended_dataset = self.synthetic_dataset_generator.extend_nxdataset([graph], "training", "final") ## TODO MAYBE CHANGE?

        if len(extended_dataset["train"][0].get_edges_ids()) > 0:
            extended_dataset.pop("test"), extended_dataset.pop("val")
            normalized_nxdatset = self.synthetic_dataset_generator.normalize_features_nxdatset(extended_dataset)
            inferred_concept_sets = self.gnns[target_concept].infer(normalized_nxdatset["train"][0],True,use_gt = False, to_sgraph = True)

            mapped_inferred_concepts = {}
            for inferred_concept in inferred_concept_sets.keys():
                self.get_logger().info(f"dbg inferred_concept {inferred_concept}")
                if inferred_concept_sets[inferred_concept]:
                    mapped_inferred_concept_sets = [set(splitting_mapping[id] for id in inferred_concept_set) for inferred_concept_set in inferred_concept_sets[inferred_concept]]
                    self.get_logger().info(f"dbg mapped_inferred_concept_sets {mapped_inferred_concept_sets}")
                    self.concept_set_trackers[inferred_concept].add_observation(mapped_inferred_concept_sets)
                    current_concept_sets, all_concept_sets = self.concept_set_trackers[inferred_concept].postprocess()
                    self.get_logger().info(f"dbg all_concept_sets {all_concept_sets}")

                mapped_inferred_concept = []
                if current_concept_sets:
                    for current_concept_set in current_concept_sets:
                                
                        hlc_id = current_concept_set[0]
                        old_llc_ids = current_concept_set[2]
                        old_llc_ids = [ id for id in old_llc_ids if id in filtered_planes_dicts_dict.keys()]
                        old_llc_ids_dict = [filtered_planes_dicts_dict[old_llc_id] for old_llc_id in old_llc_ids]

                        if len(set(old_llc_ids)) > 1:
                            concept_dict = {}
                            concept_dict["id"] = hlc_id
                            concept_dict["ws_ids"] = old_llc_ids
                            concept_dict["ws_xy_types"] = [old_llc_id_dict["xy_type"] for old_llc_id_dict in old_llc_ids_dict]
                            concept_dict["ws_msgs"] = [old_llc_id_dict["msg"] for old_llc_id_dict in old_llc_ids_dict]
                            concept_dict["center"], initial_filtered_planes_graph = self.add_hlc_node(initial_filtered_planes_graph, old_llc_ids, hlc_id, inferred_concept)
                            mapped_inferred_concept.append(concept_dict)

                mapped_inferred_concepts[inferred_concept] = mapped_inferred_concept

            # self.get_logger().info(f"dbg mapped_inferred_concepts[room] {mapped_inferred_concepts['room']}")
            if mapped_inferred_concepts and target_concept == "room":
                self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(mapped_inferred_concepts))
                [self.tmp_room_history.append(concept["center"]) for concept in mapped_inferred_concepts]

            elif mapped_inferred_concepts and target_concept == "wall":
                self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(mapped_inferred_concepts))

            elif target_concept == "RoomWall":
                if mapped_inferred_concepts["room"]:
                    self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(mapped_inferred_concepts["room"]))
                if mapped_inferred_concepts["wall"]:
                    self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(mapped_inferred_concepts["wall"]))

            
            fig = visualize_nxgraph(initial_filtered_planes_graph, image_name = f"inference HLCs to sgraph", include_node_ids= False, visualize_alone=False)
            # fig.savefig(self.report_path + "/HLC_to_sgraph.png")
            self.gnns[target_concept].metric_subplot.update_plot_with_figure(f"to Sgraph", fig, square_it = True)
            self.gnns[target_concept].metric_subplot.save(self.generation_plots_path + f"/HLC_to_sgraph_{self.generation_i}.png")
            self.generation_i += 1

        else:
            self.get_logger().info(f"Graph Reasoning: No edges in the graph!!!")


    def infer_from_rooms(self, target_concept, msg):
        if self.tmp_room_history:
            graph = GraphWrapper()
            for i, room_center in enumerate(self.tmp_room_history):
                graph.add_nodes([(i,{"type" : "room","center" : room_center, "x" : room_center,\
                                    "viz_type" : "Point", "viz_data" : room_center, "viz_feat" : 'ro'})])

            inferred_concepts = self.gnns[target_concept].cluster_floors(graph)

            # self.get_logger().info(f"flag inferred_concepts {inferred_concepts}")

    def generate_room_subgraph_msg(self, inferred_rooms):
        rooms_msg = RoomsDataMsg()
        for room_id, room in enumerate(inferred_rooms):
            x_planes, y_planes = [], []
            x_centers, y_centers = [], []
            cluster_center = []
            for plane_index, ws_type in enumerate(room["ws_xy_types"]):
                if ws_type == "x":
                    x_planes.append(room["ws_msgs"][plane_index])
                    # x_centers.append(room["ws_centers"][plane_index])
                elif ws_type == "y":
                    y_planes.append(room["ws_msgs"][plane_index])
                    # y_centers.append(room["ws_centers"][plane_index])

            if room["ws_msgs"]:

                room_msg = RoomDataMsg()
                room_msg.id = room_id
                room_msg.planes = room["ws_msgs"]
                room_msg.room_center = PoseMsg()
                room_msg.room_center.position.x = float(room["center"][0])
                room_msg.room_center.position.y = float(room["center"][1])
                room_msg.room_center.position.z = float(room["center"][2])
                rooms_msg.rooms.append(room_msg)

        return rooms_msg
    

    def add_hlc_node(self, graph, community, hlc_id, hlc_concept):
        if self.use_gnn_factors:
            max_d = 20.
            planes_centers_normalized = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in community])
            planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]]) for node_id in community]
            planes_feats_4p = np.array([self.correct_plane_direction_ndarray(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p])
            planes_feats_4p = torch.tensor(planes_feats_4p, dtype=torch.float32) if isinstance(planes_feats_4p, np.ndarray) else planes_feats_4p
            x = torch.cat((torch.tensor(planes_centers_normalized, dtype=torch.float32), 
                        planes_feats_4p[:, :3].float()), dim=1)
            zeros_row = torch.zeros(1, x.size(1), dtype=torch.float32)  # REMOVE THIS FROM F-GNN architecture
            x = torch.cat((x, zeros_row), dim=0)
            x1, x2 = [], []
            for i in range(x.size(0) - 1):
                x1.append(i)
                x2.append(x.size(0) - 1)
            edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
            batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
            nn_outputs = self.factor_nn.infer(x, edge_index, batch, "wall").numpy()[0]
            center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
        else:
            center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in community]).astype(np.float32), axis = 0)/len(community)
        
        node_id_offsets_per_concept = {"room": 100, "wall": 200}
        node_viz_feat_per_concept = {"room": 'ro', "wall": 'mo'}
        edge_viz_feat_per_concept = {"room": 'red', "wall": 'brown'}

        hlc_id_offset = hlc_id + node_id_offsets_per_concept[hlc_concept]
        graph.add_nodes([(hlc_id_offset,{"type" : hlc_concept,"viz_type" : "Point", "viz_data" : center[:2],"center" : center, "viz_feat" : node_viz_feat_per_concept[hlc_concept]})])
        
        for node_id in list(set(community)):
            graph.add_edges([(hlc_id_offset, node_id, {"type": f"ws_belongs_{hlc_concept}", "x": [], "viz_feat" : edge_viz_feat_per_concept[hlc_concept], "linewidth":1.0, "alpha":0.5})])

        return center, graph
        

    def correct_plane_direction(self,p4):
        if p4[3] > 0:
            p4 = -1 * p4
        return p4


    def generate_wall_subgraph_msg(self, inferred_walls):
        walls_msg = WallsDataMsg()
        for wall_id, wall in enumerate(inferred_walls):
            x_planes, y_planes = [], []
            x_centers, y_centers = [], []
            for plane_index, ws_type in enumerate(wall["ws_xy_types"]):
                if ws_type == "x":
                    x_planes.append(wall["ws_msgs"][plane_index])
                    # x_centers.append(wall["ws_centers"][plane_index])
                elif ws_type == "y":
                    y_planes.append(wall["ws_msgs"][plane_index])
                    # y_centers.append(wall["ws_centers"][plane_index])
            
            wall_center = wall["center"]
            # if len(x_planes) == 0 and len(y_planes) == 2:
            #     x_planes = []
            #     wall["center"] = (y_centers[0] + y_centers[1])/2
            #     wall_point = wall["center"]
            #     # wall_center = self.compute_wall_center(wall_point, y_planes)            
            #     wall_center = wall["center"]


            # elif len(x_planes) == 2 and len(y_planes) == 0:
            #     y_planes = []
            #     wall["center"] = (x_centers[0] + x_centers[1])/2
            #     wall_point = wall["center"]
            #     # wall_center = self.compute_wall_center(wall_point, x_planes)     
            #     wall_center = wall["center"]                   

            # else:
            #     x_planes, y_planes = [], []

            if x_planes or y_planes:
                wall_msg = WallDataMsg()
                wall_msg.id = wall_id
                wall_msg.x_planes = x_planes
                wall_msg.y_planes = y_planes
                wall_msg.wall_center = PoseMsg()
                wall_msg.wall_center.position.x = wall_center[0]
                wall_msg.wall_center.position.y = wall_center[1]
                wall_msg.wall_center.position.z = wall_center[2]
                wall_msg.wall_point = PointMsg()
                # wall_msg.wall_point.x = wall_point[0]
                # wall_msg.wall_point.y = wall_point[1]
                # wall_msg.wall_point.z = wall_point[2]
                
                walls_msg.walls.append(wall_msg)

        return walls_msg
    

    # def compute_wall_center(self, wall_point_inp, planes_inp):
    #     planes=copy.deepcopy(planes_inp)
    #     plane1 = planes[0]
    #     plane2 = planes[1]
    #     wall_point = copy.deepcopy(wall_point_inp)
    #     final_wall_center = self.compute_center(wall_point, plane1, plane2)

    #     return final_wall_center

       
    # def compute_infinite_room_center(self, cluster_point_inp, planes_inp):
    #     planes = copy.deepcopy(planes_inp)
    #     plane1 = planes[0]
    #     plane2 = planes[1]
    #     cluster_point = copy.deepcopy(cluster_point_inp)
    #     final_room_center = self.compute_center(cluster_point, plane1, plane2)
    #     return final_room_center
    

    # def compute_room_center(self, x_planes_inp, y_planes_inp):

    #     x_planes = copy.deepcopy(x_planes_inp)
    #     y_planes = copy.deepcopy(y_planes_inp)
    #     x_plane1 = x_planes[0]
    #     x_plane2 = x_planes[1]
        
    #     y_plane1 = y_planes[0]
    #     y_plane2 = y_planes[1]

    #     x_plane1 = self.correct_plane_direction(x_plane1)        
    #     x_plane2 = self.correct_plane_direction(x_plane2)
    #     y_plane1 = self.correct_plane_direction(y_plane1)        
    #     y_plane2 = self.correct_plane_direction(y_plane2)              

    #     vec_x, vec_y = [], []

    #     if(math.fabs(x_plane1.d) > math.fabs(x_plane2.d)):
    #         vec_x = (0.5 * (math.fabs(x_plane1.d) * np.array([x_plane1.nx, x_plane1.ny, x_plane1.nz]) - math.fabs(x_plane2.d) * np.array([x_plane2.nx, x_plane2.ny, x_plane2.nz]))) + math.fabs(x_plane2.d) * np.array([x_plane2.nx, x_plane2.ny, x_plane2.nz])
    #     else:
    #         vec_x = (0.5 * (math.fabs(x_plane2.d) * np.array([x_plane2.nx, x_plane2.ny, x_plane2.nz]) - math.fabs(x_plane1.d) * np.array([x_plane1.nx, x_plane1.ny, x_plane1.nz]))) + math.fabs(x_plane1.d) * np.array([x_plane1.nx, x_plane1.ny, x_plane1.nz])

    #     if(math.fabs(y_plane1.d) > math.fabs(y_plane2.d)):
    #         vec_y = (0.5 * (math.fabs(y_plane1.d) * np.array([y_plane1.nx, y_plane1.ny, y_plane1.nz]) - math.fabs(y_plane2.d) * np.array([y_plane2.nx, y_plane2.ny, y_plane2.nz]))) + math.fabs(y_plane2.d) * np.array([y_plane2.nx, y_plane2.ny, x_plane2.nz])
    #     else:
    #         vec_y = (0.5 * (math.fabs(y_plane2.d) * np.array([y_plane2.nx, y_plane2.ny, y_plane2.nz]) - math.fabs(y_plane1.d) * np.array([y_plane1.nx, y_plane1.ny, y_plane1.nz]))) + math.fabs(y_plane1.d) * np.array([y_plane1.nx, y_plane1.ny, y_plane1.nz])

    #     final_room_center = vec_x + vec_y
    
    #     return final_room_center


    # def compute_center(self, wall_point_inp, plane1_inp, plane2_inp):
    #     wall_point = copy.deepcopy(wall_point_inp)
    #     plane1 = copy.deepcopy(plane1_inp)
    #     plane2 = copy.deepcopy(plane2_inp)
    #     plane1 = self.correct_plane_direction(plane1)        
    #     plane2 = self.correct_plane_direction(plane2)        
        
    #     if(math.fabs(plane1.d) > math.fabs(plane2.d)):
    #         estimated_wall_center = (0.5 * (math.fabs(plane1.d) * np.array([plane1.nx, plane1.ny, plane1.nz]) - math.fabs(plane2.d) *  np.array([plane2.nx, plane2.ny, plane2.nz]))) + math.fabs(plane2.d) *  np.array([plane2.nx, plane2.ny, plane2.nz])
    #     else:
    #         estimated_wall_center = (0.5 * (math.fabs(plane2.d) * np.array([plane2.nx, plane2.ny, plane2.nz]) - math.fabs(plane1.d) * np.array([plane1.nx, plane1.ny, plane1.nz]))) + math.fabs(plane1.d) * np.array([plane1.nx, plane1.ny, plane1.nz])

    #     estimated_wall_center_normalized = estimated_wall_center[:3] / np.linalg.norm(estimated_wall_center)
    #     final_wall_center =  estimated_wall_center[:3] + (wall_point -  np.dot(wall_point, estimated_wall_center_normalized) * estimated_wall_center_normalized)

    #     return final_wall_center       


    def correct_plane_direction_msg(self, plane):
        if(plane.d > 0):
            plane.nx = -1 * plane.nx
            plane.ny = -1 * plane.ny
            plane.nz = -1 * plane.nz
            plane.d = -1 * plane.d
        
        return plane 
    
    def correct_plane_direction_ndarray(self,p4):
        if p4[3] > 0:
            p4 = -1 * p4
        return p4
    
    def characterize_ws(self, points):
        points = np.array([np.array([point.x,point.y,0]) for point in points])
        if len(points) > 0:
            four_points = [points[np.argmax(points[:,0])],points[np.argmin(points[:,0])],points[np.argmax(points[:,1])],points[np.argmin(points[:,1])]] 
            max_dist = 0
            for i, point_1 in enumerate(four_points):
                points_2 = copy.deepcopy(four_points)
                points_2.reverse()
                for point_2 in points_2:
                    dist = abs(np.linalg.norm(point_1 - point_2))
                    if dist > max_dist:
                        max_dist = dist
                        limit_1 = point_1
                        limit_2 = point_2
                        center = limit_2/2 + limit_1/2
            return center, [limit_1, limit_2], max_dist
        else:
            return [], [], []
    

    def filter_overlapped_ws(self, planes_dict):
        # self.get_logger().info(f"Graph Reasoning: filter overlapped wall surfaces")
        segments = [ plane_dict["segment"] for plane_dict in planes_dict]
        expansion = 0.1
        coverage_thr = 0.6

        def augment_segment(segment):
            norm = (segment[0] - segment[1])/abs(np.linalg.norm(segment[0] - segment[1]))
            ort_norm = np.concatenate([np.squeeze(np.rot90([norm[:2]])), [0.]])
            rectangle = Polygon([segment[0]+(norm+ort_norm)*expansion, segment[0]+(norm-ort_norm)*expansion,
                         segment[1]-(norm+ort_norm)*expansion, segment[1]-(norm-ort_norm)*expansion])
            return rectangle
        agumented_segments = [ augment_segment(segment) for segment in segments]

        def compute_coverage(rectangle_1, rectangle_2):
            intersection = rectangle_1.intersection(rectangle_2)
            coverage = intersection.area / rectangle_1.area
            return coverage

        filterout_planes_index = []
        for i, agumented_segment in enumerate(agumented_segments):
            for jj in range(len(agumented_segments) - i - 1):
                j = len(agumented_segments) - jj - 1
                if j not in filterout_planes_index and (compute_coverage(agumented_segment, agumented_segments[j]) > coverage_thr):
                    filterout_planes_index.append(i)
                    break

        filteredin_planes_dict = copy.deepcopy(planes_dict)
        filterout_planes_index.reverse()
        [filteredin_planes_dict.pop(i) for i in filterout_planes_index]

        return filteredin_planes_dict
            

    def split_ws(self, planes_dict):
        # self.get_logger().info(f"Graph Reasoning: splitting wall surfaces")
        extension = 1.
        thr_length = 0.3
        all_extended_segments = []
        current_id = 0
        new_planes_dicts = []
        for plane_dict in planes_dict:
            # extend segment
            segment = plane_dict["segment"]
            norm = (segment[0] - segment[1])/abs(np.linalg.norm(segment[0] - segment[1]))
            plane_dict["extended_segment"] = [segment[0] + norm*extension, segment[1] - norm*extension]
            all_extended_segments.append(plane_dict["extended_segment"])

        for i, plane_dict in enumerate(planes_dict):
            segment = plane_dict["segment"]
            rest_segments = copy.deepcopy(all_extended_segments)
            rest_segments.pop(i)

            intersections = []
            distances_to_1 = []
            for other_segment in rest_segments:
                if segments_distance(segment, other_segment) == 0.0:
                    intersections.append(segment_intersection(segment, other_segment))
                    distances_to_1.append(abs(np.linalg.norm(intersections[-1] - segment[0])))

            if intersections:
                new_segments = []
                index_sorted = np.argsort(distances_to_1)
                for j,k in enumerate(index_sorted):
                    if j == 0:
                        new_segments.append([segment[0], intersections[k]])
                    if j < len(intersections) - 1:
                        new_segments.append([intersections[k], intersections[index_sorted[j+1]]])
                    else:
                        new_segments.append([intersections[k], segment[1]])

                for new_segment in new_segments:
                    length = abs(np.linalg.norm(new_segment[0] - new_segment[1]))
                    if length > thr_length:
                        new_plane_dict = {"old_id": plane_dict["id"], "id": current_id, "segment": new_segment, "normal": plane_dict["normal"], "length": length, "xy_type": plane_dict["xy_type"], "msg": plane_dict["msg"]}
                        current_id += 1
                        new_plane_dict["center"] = new_segment[0]/2 + new_segment[1]/2
                        new_planes_dicts.append(new_plane_dict)

            else:
                length = abs(np.linalg.norm(plane_dict["segment"][0] - plane_dict["segment"][1]))
                if length > 0.5:
                    new_plane_dict = copy.deepcopy(plane_dict)
                    new_plane_dict["old_id"] = plane_dict["id"]
                    new_plane_dict["id"] = current_id
                    current_id += 1
                    new_planes_dicts.append(new_plane_dict)
        return new_planes_dicts
    
    def dbg_fake_plane_msgs(self):
        class PlanePointFake():
            def __init__(self_fake):
                self_fake.x=0.0
                self_fake.y=0.0
                self_fake.z=0.0

        class PlaneMsgFake():
            def __init__(self_fake):
                self_fake.plane_points = [PlanePointFake(),PlanePointFake()]
                self_fake.id = 0
                self_fake.nx = 0.0
                self_fake.ny = 0.0
                self_fake.nz = 0.0
                self_fake.d = None

        d = 8
        w = 0.3

        plane_msgs = []
        plane_msg = PlaneMsgFake()
        plane_msg.id = 0
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=0.0
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 1
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=d
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=-1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 2
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=0.0
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 3
        plane_msg.plane_points[0].x=d
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=-1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 4
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=0.0
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 5
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=d
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=-1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 6
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=0.0 + d + w
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 7
        plane_msg.plane_points[0].x=d + d + w
        plane_msg.plane_points[0].y=0.0
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=d
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=-1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 8
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=0.0 + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 9
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=d + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=-1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 10
        plane_msg.plane_points[0].x=0.0
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=0.0
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 11
        plane_msg.plane_points[0].x=d
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=-1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 12
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=0.0 + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 13
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=d + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=0.0
        plane_msg.ny=-1.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 14
        plane_msg.plane_points[0].x=0.0 + d + w
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=0.0 + d + w
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        plane_msg = PlaneMsgFake()
        plane_msg.id = 15
        plane_msg.plane_points[0].x=d + d + w
        plane_msg.plane_points[0].y=0.0 + d + w
        plane_msg.plane_points[0].z=0.0

        plane_msg.plane_points[1].x=d + d + w
        plane_msg.plane_points[1].y=d + d + w
        plane_msg.plane_points[1].z=0.0

        plane_msg.nx=-1.0
        plane_msg.ny=0.0
        plane_msg.nz=0.0

        plane_msgs.append(plane_msg)

        return plane_msgs

    
    def parse_arguments(self, args):
        parser = argparse.ArgumentParser(description='Process some strings.')
        parser.add_argument('--generated_entities', type=str, default='[]',
                            help='A list of strings')
        parser.add_argument('--use_gnn_factors', type=bool, default=True,
                            help='Use a GNNs as the factors')
        parser.add_argument('--log_path', type=str, default='.',
                            help='Experiment log path')
        args, unknown = parser.parse_known_args()

        args.generated_entities = ast.literal_eval(args.generated_entities)
        return args


def main(args=None):
    rclpy.init(args=args)

    graph_matching_node = GraphReasoningNode(args)

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
