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
from graph_datasets.graph_visualizer_interactive import visualize_nxgraph_interactive

from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.config import get_config as datasets_get_config
from graph_matching.utils import segments_distance, segment_intersection, plane_6_params_to_4_params
from graph_factor_nn.FactorNNBridge import FactorNNBridge

import math

graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph

class GenerationGTNode(Node):
    def __init__(self, args):
        super().__init__('generation_gt')

        args = self.parse_arguments(args)
        
        # args = ["room", "wall"]
        self.get_logger().info(f"dbg args {args}")
        self.find_rooms, self.find_walls, self.find_floors, self.find_RoomWall = False, False, False, False
        
        self.use_gnn_factors = args.use_gnn_factors
        if self.use_gnn_factors:
            self.factor_nn = FactorNNBridge(["room", "wall", "floor"])

        # dataset_ref = args.dataset_ref.split("-")
        # self.GTs = datasets_get_config("dataset_GT_configs")[dataset_ref[0]][dataset_ref[1]]

        if "room" in args.generated_entities:
            self.find_rooms = True
        if "wall" in args.generated_entities:
            self.find_walls = True
        if "floor" in args.generated_entities:
            self.find_floors = True
        if "RoomWall" in args.generated_entities:
            self.find_RoomWall = True

        self.last_planes_msg = []

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

        self.synthetic_dataset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger(), self.report_path)
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")
        self.node_start_time = time.perf_counter()
        self.first_room_detected = False
        self.tmp_room_history = []
        self.generation_times_history = []
        self.hlc_id_counters = {"room" : 1000, "wall" : 2000}

    #     self.process_planes_loop()    

    # def process_planes_loop(self):
    #     self.get_logger().info(f"flag")
    #     try:
    #         while rclpy.ok():
    #             self.get_logger().info(f"flag2")
    #             if self.last_planes_msg:
    #                 self.get_logger().info(f"flag3")
    #                 self.infer_from_planes(self.last_planes_msg)
    #                 self.last_planes_msg = []
    #     except KeyboardInterrupt:
    #         pass
    #     finally:
    #         self.destroy_node()
    #         rclpy.shutdown()

  

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
        self.last_planes_msg = msg
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
        target_concept = "RoomWall"
        
        if len(msg.x_planes) == 0 or len(msg.y_planes) == 0:
            return
        
        input_planes_ids = []
        planes_dicts = {}
        for i, plane_msg in enumerate(msg.x_planes + msg.y_planes):
            if len(plane_msg.plane_points) != 0:
                input_planes_ids.append(plane_msg.id)
                plane_dict = {"id": plane_msg.id, "normal" : np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])}
                plane_dict["xy_type"] = "x" if i<len(msg.x_planes) else "y"
                plane_dict["msg"] = plane_msg
                plane_dict["center"], plane_dict["segment"], plane_dict["length"] = self.characterize_ws(plane_msg.plane_points)
                planes_dicts[plane_msg.id] = plane_dict
        input_planes_ids.sort()

        plane_mapping = {}
        all_planes_ids = []
        for i, plane_id in enumerate(input_planes_ids):
            all_planes_ids.append(i)
            plane_mapping[i] = plane_id

        initial_graph = GraphWrapper()
        initial_graph.to_directed()
        for plane_id in all_planes_ids:
            plane_dict = planes_dicts[plane_mapping[plane_id]]
            initial_graph.add_nodes([(plane_id,{"type" : "ws","center" : plane_dict["center"], "label": 1, "normal" : plane_dict["normal"],\
                                    "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
                                    "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
        # fig = visualize_nxgraph(initial_graph, image_name = f"input from sgraphs", include_node_ids= True, visualize_alone=False)
        # fig.savefig(self.generation_plots_path + f"/planes_from_sgraph_{self.generation_i}.png")

        observed_gt_hlcs_dict = visualize_nxgraph_interactive(initial_graph, "set GT from planes", visualize_alone=True, logger=self.get_logger())
        
        inferred_concepts = {}

        self.get_logger().info(f"dbg observed_gt_hlcs_dict {observed_gt_hlcs_dict}")
        inferred_concept_mapping = {"R":"room", "r":"room", "W":"wall","w":"wall"}
        concept_dicts = []
        for key in observed_gt_hlcs_dict.keys():
            for llc_list in observed_gt_hlcs_dict[key]:
                self.hlc_id_counters[inferred_concept_mapping[key]] += 1
                hlc_id = self.hlc_id_counters[inferred_concept_mapping[key]]
                concept_dict = {}
                concept_dict["id"] = int(self.hlc_id_counters[inferred_concept_mapping[key]])
                concept_dict["ws_ids"] = llc_list
                concept_dict["ws_xy_types"] = [planes_dicts[plane_mapping[llc_id]]["xy_type"] for llc_id in llc_list]
                concept_dict["ws_msgs"] = [planes_dicts[plane_mapping[llc_id]]["msg"] for llc_id in llc_list]
                concept_dict["center"], initial_graph = self.add_hlc_node(initial_graph, llc_list, int(hlc_id), inferred_concept_mapping[key])
                concept_dicts.append(concept_dict)

            inferred_concepts[inferred_concept_mapping[key]] = concept_dicts
        
        if target_concept == "RoomWall":
            if "room" in inferred_concepts.keys() and inferred_concepts["room"]:
                self.get_logger().info(f'dbg self.generate_room_subgraph_msg(inferred_concepts["room"]) {self.generate_room_subgraph_msg(inferred_concepts["room"])}')
                self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(inferred_concepts["room"]))
            if "wall" in inferred_concepts.keys() and inferred_concepts["wall"]:
                self.get_logger().info(f'dbg self.generate_wall_subgraph_msg(inferred_concepts["wall"]) {self.generate_wall_subgraph_msg(inferred_concepts["wall"])}')
                self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(inferred_concepts["wall"]))

        fig = visualize_nxgraph(initial_graph, image_name = f"GT HLCs to sgraph", include_node_ids= True, visualize_alone=False)
        fig.savefig(self.generation_plots_path + f"/HLC_to_sgraph_{self.generation_i}.png")
        self.generation_i += 1

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

    def correct_plane_direction_ndarray(self,p4):
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
    
    
    def parse_arguments(self, args):
        parser = argparse.ArgumentParser(description='Process some strings.')
        parser.add_argument('--generated_entities', type=str, default='[]',
                            help='A list of strings')
        parser.add_argument('--use_gnn_factors', type=bool, default=True,
                            help='Use a GNNs as the factors')
        parser.add_argument('--log_path', type=str, default='.',
                            help='Experiment log path')
        parser.add_argument('--dataset_ref', type=str, default='.',
                            help='Type and tag of datset')
        args, unknown = parser.parse_known_args()

        args.generated_entities = ast.literal_eval(args.generated_entities)
        return args


def main(args=None):
    rclpy.init(args=args)

    graph_reasoning_node = GenerationGTNode(args)

    rclpy.spin(graph_reasoning_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_reasoning_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
