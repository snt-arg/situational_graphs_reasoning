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
import time, os, json, shutil, sys, torch, csv
import copy
import numpy as np
import ament_index_python
import argparse
import ast
import struct
import matplotlib.colors as mcolors
from rclpy.node import Node
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
import open3d as o3d
from rclpy.time import Time
import math

# from tf2_ros.transform_listener import TransformListener
# from tf2_ros.buffer import Buffer
# from tf2_ros.buffer_interface import BufferInterface
# import tf2_geometry_msgs 
# from visualization_msgs.msg import Marker as MarkerMsg
from geometry_msgs.msg import Pose as PoseMsg
# from geometry_msgs.msg import Vector3 as Vector3Msg
# from geometry_msgs.msg import PointStamped as PointStampedMsg
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Quaternion as QuaternionMsg
# from geometry_msgs.msg import Transform as TransformMsg
# from geometry_msgs.msg import TransformStamped as TransformStampedMsg
from std_msgs.msg import ColorRGBA as ColorRGBSMsg
from std_msgs.msg import Header as HeaderMsg
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from rclpy.parameter import Parameter
# from rclpy.parameter import ParameterType
# from ament_index_python.packages import get_package_share_directory
from shapely.geometry import Polygon
from tf_transformations import (
    quaternion_from_euler,   # build extra rotation
    quaternion_multiply      # combine quaternions
)

from situational_graphs_msgs.msg import PlanesData as PlanesDataMsg
from situational_graphs_msgs.msg import PlaneData as PlaneDataMsg
from situational_graphs_msgs.msg import RoomsData as RoomsDataMsg
from situational_graphs_msgs.msg import RoomData as RoomDataMsg
from situational_graphs_msgs.msg import WallsData as WallsDataMsg
from situational_graphs_msgs.msg import WallData as WallDataMsg
from situational_graphs_msgs.srv import RemoveRoom as RemoveRoomSrv
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg
from visualization_msgs.msg import Marker as MarkerMsg
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import TransformStamped as TransformStampedMsg
from geometry_msgs.msg import Point as PointMsg
from situational_graphs_reasoning_msgs.msg import Graph as GraphMsg

from graph_reasoning.GNNWrapper import GNNWrapper
from graph_reasoning.EvolvingSetsTracker import EvolvingSetsTracker
from graph_reasoning.config import get_config as reasoning_get_config
from graph_reasoning.pths import get_pth as reasoning_get_pth
from graph_reasoning.IncrementalVideoUpdater import IncrementalVideoUpdater
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.config import get_config as datasets_get_config
from graph_matching.utils import segments_distance, segment_intersection, plane_6_params_to_4_params

from graph_factor_nn.FactorNNBridge import FactorNNBridge
from graph_factor_nn.FactorNN import FactorNN

graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph


import numpy as np
from scipy.spatial.transform import Rotation                          # pip install scipy
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import rclpy, tf2_ros
from rclpy.duration import Duration
from tf_transformations import quaternion_matrix   # sudo apt install python3-tf-transformations
def transform_pointcloud2(cloud_in: PointCloud2,
                          tf_msg,                     # geometry_msgs/TransformStamped
                          target_frame: str = 'map') -> PointCloud2:
    """
    Apply the 6-DoF transform in `tf_msg` to every point in `cloud_in`
    and return a new PointCloud2 stamped in `target_frame`.

    Keeps all extra fields (RGB, intensity, etc.) unmodified.
    """
    # 1. Build a 4×4 homogeneous matrix from the TransformStamped
    trans = np.array([tf_msg.transform.translation.x,
                      tf_msg.transform.translation.y,
                      tf_msg.transform.translation.z])
    quat  = [tf_msg.transform.rotation.x,
             tf_msg.transform.rotation.y,
             tf_msg.transform.rotation.z,
             tf_msg.transform.rotation.w]

    T = quaternion_matrix(quat)        # rotation → 4×4
    T[0:3, 3] = trans                  # add translation

    # 2. Read every point.  keep all remaining channels (*rest)
    pts_iter = pc2.read_points(cloud_in, skip_nans=False)
    transformed = []

    for pt in pts_iter:
        x, y, z, *rest = pt
        xyz1 = np.array([x, y, z, 1.0])
        x_m, y_m, z_m, _ = T @ xyz1
        transformed.append((x_m, y_m, z_m, *rest))

    # 3. Re-pack into a PointCloud2
    cloud_out = pc2.create_cloud(
        cloud_in.header,               # copy original header / fields layout
        cloud_in.fields,
        transformed)

    cloud_out.header.frame_id = target_frame
    return cloud_out


class GraphReasoningNode(Node):
    def __init__(self, args):
        super().__init__('graph_reasoning')

        args = self.parse_arguments(args)
        
        # args = ["room", "wall"]
        self.get_logger().info(f"dbg args {args}")
        self.find_rooms, self.find_walls, self.find_floors, self.find_RoomWall = False, False, False, False
        
        self.use_gnn_factors = args.use_gnn_factors
        if self.use_gnn_factors:
            self.factor_nn_bridges = FactorNNBridge(["room_msd", "room_naive", "wall_naive", "floor"])
            config_path = "/home/adminpc/workspaces/reasoning_ws/src/graph_factor_nn"
            with open(os.path.join(config_path, f"config/room.json")) as f:
                config = json.load(f)

            self.factor_nn_objects = FactorNN(config, None, None, args.log_path)
            self.factor_nn_objects.load_model(config_path + "/pths/room_msd.pth")

        self.ablations=eval(args.ablations)

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
            # self.concept_set_trackers["room"] = EvolvingSetsTracker(logger = self.get_logger())
            # self.concept_set_trackers["wall"] = EvolvingSetsTracker(logger = self.get_logger())
            self.concept_set_trackers["room"] = EvolvingSetsTracker()
            self.concept_set_trackers["wall"] = EvolvingSetsTracker()

        self.generation_plots_path = args.log_path + "/generation_plots"
        self.order_time_log_path = args.log_path + "/order_time_log.csv"
        os.makedirs(self.generation_plots_path)
        self.generation_i = 0
        self.colors = list(mcolors.XKCD_COLORS.values())[:30]
        self.v_sgraphs_planes_dict = {}

        # self.graph_reasoning_rooms_settings = reasoning_get_config("same_room_best")
        # self.graph_reasoning_walls_settings = reasoning_get_config("same_wall_best")
        # self.graph_reasoning_floors_settings = reasoning_get_config("same_floor_training")
        self.graph_reasoning_RoomWall_settings = reasoning_get_config("same_RoomWall_best")
        self.reasoning_package_path = ament_index_python.get_package_share_directory("graph_reasoning")

        dataset_settings = datasets_get_config("graph_reasoning")

        dataset_settings["training_split"]["val"] = 0.0
        dataset_settings["training_split"]["test"] = 0.0
        
        self.dataset_settings = dataset_settings
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

            # self.gnns["RoomWall"].save_model(os.path.join(self.report_path,"model_RoomWall_best.pth"))

        self.synthetic_dataset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger(), self.report_path)
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")
        self.node_start_time = time.perf_counter()
        self.first_room_detected = False
        self.planes_dicts = None
        self.current_concept_sets = {}
        self.generation_times_history = []
        self.video_updater = IncrementalVideoUpdater(output_filename=self.generation_plots_path + f"/HLC_to_sgraph.avi", fps=0.5, logger=self.get_logger())
        self.video_updater.start()

        wait_for_TFs = False

        if wait_for_TFs:
            # --- TF setup -------------------------------------------------------
            self.tf_buf      = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

            # --- BLOCK here until the transform shows up -----------------------
            while not self.tf_buf.can_transform(
                    'map', 'plane', Time(),           # Time() == “latest”
                    timeout=Duration(seconds=0.1)):
                self.get_logger().info('Waiting for map → plane TF …')
                rclpy.spin_once(self, timeout_sec=0.5)   # let TF msgs flow
            self.get_logger().info('map → plane TF is available')
            # --- Got it: look it up once and continue --------------------------
            try:
                self.plane_to_map = self.tf_buf.lookup_transform(
                    'map', 'plane', Time())            # latest transform
            except (tf2_py.LookupException,
                    tf2_py.ExtrapolationException) as e:
                self.get_logger().fatal(f'TF lookup failed: {e}')
                raise RuntimeError('Unexpected TF failure') from e

            # choose the extra rotation you want to apply (example: +10 deg yaw)
            yaw_offset_deg = 0.0
            yaw_offset_rad = math.radians(yaw_offset_deg)

            # build the “offset” quaternion
            #   (roll, pitch, yaw) = (0, 0, yaw_offset_rad)  → rotate about +Z
            q_offset = quaternion_from_euler(0.0, 0.0, yaw_offset_rad)   # (x,y,z,w)

            # current rotation in the TransformStamped
            q_orig = [
                self.plane_to_map.transform.rotation.x,
                self.plane_to_map.transform.rotation.y,
                self.plane_to_map.transform.rotation.z,
                self.plane_to_map.transform.rotation.w,
            ]

            # multiply:  q_new = q_offset ⊗ q_orig
            q_new = quaternion_multiply(q_offset, q_orig)

            # write it back into the TransformStamped
            self.plane_to_map.transform.rotation.x = q_new[0]
            self.plane_to_map.transform.rotation.y = q_new[1]
            self.plane_to_map.transform.rotation.z = q_new[2]
            self.plane_to_map.transform.rotation.w = q_new[3]
       

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

        ### Order Time Log
        self.get_logger().info(f"dbg order_time_log_path {self.order_time_log_path}")
        # f = open(self.order_time_log_path, mode="a", buffering=1024*1024, newline="")
        # self.order_time_log_writer = csv.writer(f)

        self.order_time_log_file = open(
            self.order_time_log_path, mode="a", buffering=1024*1024, newline=""
        )
        self.order_time_log_writer = csv.DictWriter(
            self.order_time_log_file, fieldnames=["order", "times","n_found_concepts"]
        )

    def set_interface(self):
        self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_all_planes_callback, 10)
        self.create_subscription(MarkerArrayMsg,'/s_graphs/markers', self.s_graph_room_marker_callback, 10)
        self.create_subscription(GraphMsg,'/s_graphs/graph_structure', self.s_graph_structure_callback, 1)
        self.create_subscription(MarkerArrayMsg,'/orb_slam3/plane_labels', self.orb_slam3_plane_labels_callback, 1)
        
        qos_latest = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(PointCloud2Msg,'/orb_slam3/plane_point_clouds', self.orb_slam3_plane_point_pointclouds_callback, qos_latest)

        self.wall_subgraph_publisher = self.create_publisher(WallsDataMsg, '/wall_segmentation/wall_data', 10)
        self.room_subgraph_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)
        self.v_sgraphs_markers_publisher = self.create_publisher(MarkerArrayMsg, '/aaaaaaa', 10)
        # self.floor_subgraph_publisher = self.create_publisher(FloorDataMsg, '/floor_plan/floor_data', 10)

        # _ = self.create_timer(2.0, self.infer_from_planes)

        self.remove_room_client = self.create_client(RemoveRoomSrv, '/s_graphs/remove_room')



    def s_graph_all_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in ALL planes topic")
        self.infer_from_planes_lidar(msg)
        # end_time = time.time()
        # self.generation_times_history.append(end_time - start_time)
        # averaged_generation_times_history = sum(self.generation_times_history)/len(self.generation_times_history)

    def s_graph_last_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in LAST planes topic")
        self.infer_from_planes_lidar("room", msg)

    def s_graph_room_marker_callback(self, msg):
        
        if self.find_floors:
            self.infer_from_rooms("floor", msg)

    def s_graph_structure_callback(self, msg):
        concept_names_mapping = {"Room": "room", "Wall": "wall", "Plane": "ws"}
        s_graph = GraphWrapper({"name" : msg.name, "nodes":[], "edges":[]})
        for node in msg.nodes:
            s_graph.add_nodes([(node.id,{"type" : concept_names_mapping[node.type]})])
        for edge in msg.edges:
            s_graph.add_edges([(edge.origin_node, edge.target_node, {})])
        
        concepts_in_sgraphs = {}
        for concept_name in ["room", "wall"]:
            concepts_list = []
            for concept_id in list(s_graph.filter_graph_by_node_types(concept_name).get_nodes_ids()):
                concept_tuple = (concept_id, list(s_graph.get_neighbourhood_graph(concept_id).filter_graph_by_node_types("ws").get_nodes_ids()))
                concepts_list.append(concept_tuple)
            concepts_in_sgraphs[concept_name] = concepts_list
        
        for concept_name in self.current_concept_sets.keys():
            for concept_list_sgraph in concepts_in_sgraphs[concept_name]:
                sgraph_concept_id = concept_list_sgraph[0]
                found = False
                for concept_list_generation in self.current_concept_sets[concept_name]:
                    if set(concept_list_sgraph[1]) == set(concept_list_generation[2]):
                        found = True
                    
                if not found and concept_name == "room" and "room_removal" not in self.ablations:
                    self.remove_room_from_sgraphs(sgraph_concept_id)


            # ## Remove outdated mapped concepts
            # for map_key in copy.deepcopy(list(self.s_graph_concepts_map[concept_name].keys())):
            #     sgraph_concept_ids = [concept_tuple[0] for concept_tuple in concepts_in_sgraphs[concept_name]]
            #     if self.s_graph_concepts_map[concept_name][map_key] not in sgraph_concept_ids:
            #         self.s_graph_concepts_map[concept_name].pop(map_key, None)

            # ##remove concepts not in current generation
            # if concept_name in self.concepts_to_remove_sgraphs.keys() and self.concepts_to_remove_sgraphs[concept_name] and concept_name == "room":
            #     for id_to_remove in self.concepts_to_remove_sgraphs[concept_name]:
            #         if id_to_remove in self.s_graph_concepts_map[concept_name].values():
            #             self.remove_room_from_sgraphs(id_to_remove)
            #             self.concepts_to_remove_sgraphs[concept_name] =  list(filter((id_to_remove).__ne__, self.concepts_to_remove_sgraphs[concept_name]))

    def orb_slam3_plane_labels_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.markers)} planes received in orb_slam3 plane labels topic")
        
        for marker in msg.markers:
            marker_dict = {}
            if marker.ns == "planeLabel":
                marker_dict["text"] = marker.text
                marker_dict["color"] = marker.color
                marker_dict["xy_type"] = "x"
                id = marker_dict["text"].split("#")[1]
                marker_dict["id"] = id

                if id not in self.v_sgraphs_planes_dict.keys():
                    self.v_sgraphs_planes_dict[id] = marker_dict
                else:
                    self.v_sgraphs_planes_dict[id]["text"] = marker_dict["text"]
                    self.v_sgraphs_planes_dict[id]["color"] = marker.color
                    self.v_sgraphs_planes_dict[id]["xy_type"] = marker_dict["xy_type"]
                    self.v_sgraphs_planes_dict[id]["id"] = marker_dict["id"]

            elif marker.ns == "planeNormal":
                p1 = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z], dtype=float)
                p2 = np.array([marker.points[1].x, marker.points[1].y, marker.points[1].z], dtype=float)
                v = p2 - p1
                q = [self.plane_to_map.transform.rotation.x,
                    self.plane_to_map.transform.rotation.y,
                    self.plane_to_map.transform.rotation.z,
                    self.plane_to_map.transform.rotation.w]

                # 2.  4×4 homogeneous matrix → take the 3×3 rotation block
                R = quaternion_matrix(q)[:3, :3]    # slice keeps only rotation

                # 3.  Rotate the vector (ignore translation)
                v_map = R @ v 
                normal = v_map / np.linalg.norm(v_map)
                if abs(normal[2]) < 0.8:
                    for plane_dict_key in self.v_sgraphs_planes_dict.keys():
                        if self.v_sgraphs_planes_dict[plane_dict_key]["color"] == marker.color:
                            cos_theta = v[2] / np.linalg.norm(v)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)

                            self.v_sgraphs_planes_dict[plane_dict_key]["normal"] = normal
                
                                       
    def orb_slam3_plane_point_pointclouds_callback(self, cloud):
        if self.plane_to_map:
            cloud_map = transform_pointcloud2(
                cloud,           # PointCloud2 in frame "plane"
                self.plane_to_map)
            buckets = self.split_to_vector3_lists_by_color(cloud_map)
            # self.to_open3d(buckets)

            for (r, g, b), points in buckets.items():
            
                for plane_dict_key in self.v_sgraphs_planes_dict.keys():
                    color2 = self.v_sgraphs_planes_dict[plane_dict_key]["color"]
                    r2, g2, b2 = int(round(color2.r * 255)), int(round(color2.g * 255)), int(round(color2.b * 255))
                    distance = math.sqrt((r - r2)**2 + (g - g2)**2 + (b - b2)**2)
                    if distance < 1:
                        center, segment, length = self.characterize_ws(points)
                        self.v_sgraphs_planes_dict[plane_dict_key]["center"] = center
                        self.v_sgraphs_planes_dict[plane_dict_key]["segment"] = segment
                        self.v_sgraphs_planes_dict[plane_dict_key]["length"] = length

                        class fake_msg:
                            def __init__(self):
                                self.d = 0
                        plane_msg = PlaneDataMsg()
                        plane_msg.d = 0.0

                        self.v_sgraphs_planes_dict[plane_dict_key]["msg"] = plane_msg

            complete_planes_dicts = []
            for plane_dict_key in self.v_sgraphs_planes_dict.keys():
                if "center" in self.v_sgraphs_planes_dict[plane_dict_key].keys() and "normal" in self.v_sgraphs_planes_dict[plane_dict_key].keys():
                    if self.v_sgraphs_planes_dict[plane_dict_key] not in complete_planes_dicts:
                        complete_planes_dicts.append(self.v_sgraphs_planes_dict[plane_dict_key])


            if len(complete_planes_dicts) > 1:
                self.planes_dicts = complete_planes_dicts
            else:
                self.planes_dicts = None


    def infer_from_planes_lidar(self, msg):
        if len(msg.x_planes) == 0 or len(msg.y_planes) == 0:
            return
        
        planes_msgs = msg.x_planes + msg.y_planes
        planes_dicts = []
        for i, plane_msg in enumerate(planes_msgs):
            if len(plane_msg.plane_points) != 0:
                plane_dict = {"id": plane_msg.id, "normal" : np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])}
                plane_dict["xy_type"] = "x" if i<len(msg.x_planes) else "y"
                plane_dict["msg"] = plane_msg
                plane_dict["center"], plane_dict["segment"], plane_dict["length"] = self.characterize_ws(plane_msg.plane_points)
                planes_dicts.append(plane_dict)

        self.planes_dicts = planes_dicts
        self.infer_from_planes()


    def infer_from_planes(self):
        self.get_logger().info(f"starting infer_from_planes")

        if self.planes_dicts == None:
            self.get_logger().info(f"There are no stored planes")
            return
        else:
            planes_dicts = copy.deepcopy(self.planes_dicts)
            self.planes_dicts = None

        times = {"start": self.get_clock().now().nanoseconds // 1_000_000}
        target_concept = "RoomWall"
        
        graph = GraphWrapper()
        graph.to_directed()
        initial_filtered_planes_graph = GraphWrapper()
        initial_filtered_planes_graph.to_directed()

        # ## Degug
        # initial_planes_graph = GraphWrapper()
        # initial_planes_graph.to_directed()
        # for plane_dict in planes_dicts:
        #     initial_planes_graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "label": 1, "normal" : plane_dict["normal"],\
        #                             "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
        #                             "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
        # fig = visualize_nxgraph(initial_planes_graph, image_name = f"filtered input from sgraphs", include_node_ids= True, visualize_alone=False)
        # fig.savefig(self.generation_plots_path + f"/input_from_sgraph_{self.generation_i}.png")

        # ## Debug End
        filtered_planes_dicts = self.filter_overlapped_ws(planes_dicts)
        filtered_planes_dicts_dict = {plane_dict["id"]: plane_dict for plane_dict in filtered_planes_dicts}
        for plane_dict in filtered_planes_dicts:
            initial_filtered_planes_graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "label": 1, "normal" : plane_dict["normal"],\
                                    "viz": {"type" : "Line", "limits" : plane_dict["segment"],"center" : plane_dict["center"], "feat" : "black"},\
                                    "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
        # fig = visualize_nxgraph(initial_filtered_planes_graph, image_name = f"filtered input from sgraphs", include_node_ids= True, visualize_alone=False)
        # fig.savefig(self.generation_plots_path + f"/initial_filtered_planes_graph_{self.generation_i}.png")
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
                                           "viz":{"type" : "Line", "limits" : plane_dict["segment"],"center" : plane_dict["center"], "feat" : "black"},\
                                           "linewidth": 2.0, "limits": plane_dict["segment"], "d" : plane_dict["msg"].d})])
            splitting_mapping[plane_dict["id"]] = plane_dict["old_id"]
        graph_to_sgraphs = copy.deepcopy(initial_filtered_planes_graph)

        # Inference
        prox_graph_order = copy.deepcopy(graph.graph.number_of_nodes())
        graph.to_directed()
        extended_dataset = self.synthetic_dataset_generator.extend_nxdataset([graph], "training", "final") ## TODO MAYBE CHANGE?
        times["preprocessing"] = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]

        if len(extended_dataset["train"][0].get_edges_ids()) > 0:
            extended_dataset.pop("test"), extended_dataset.pop("val")
            normalized_nxdatset = self.synthetic_dataset_generator.normalize_features_nxdatset(extended_dataset)
            self.gnns[target_concept].set_nxdataset(normalized_nxdatset, None)
            self.gnns[target_concept].visualize_hetero_features("train")
            use_mc_entropy = "use_mc_entropy" not in self.ablations
            inferred_concept_sets = self.gnns[target_concept].infer(normalized_nxdatset["train"][0],True,use_gt = False, to_sgraph = True, use_mc_entropy = use_mc_entropy)
            times["sem_gat_inference"] = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]
            mapped_inferred_concepts = {}
            for inferred_concept in inferred_concept_sets.keys():
                if isinstance(inferred_concept_sets[inferred_concept], list): 
                    mapped_inferred_concept_sets = [set(splitting_mapping[id] for id in inferred_concept_set) for inferred_concept_set in inferred_concept_sets[inferred_concept]]
                    self.concept_set_trackers[inferred_concept].add_observation(mapped_inferred_concept_sets)
                    self.current_concept_sets[inferred_concept], all_concept_sets = self.concept_set_trackers[inferred_concept].postprocess()

                else:
                    self.current_concept_sets[inferred_concept] = []

            times["time_stabilization"] = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]
            n_found_concepts = sum(len(v) for v in self.current_concept_sets.values())

            ### MET GNN
            for inferred_concept in inferred_concept_sets.keys():
                mapped_inferred_concept = []
                if self.current_concept_sets[inferred_concept]:
                    for current_concept_set in self.current_concept_sets[inferred_concept]:
                        hlc_id = current_concept_set[0]
                        semantic_confidence = current_concept_set[1]
                        old_llc_ids = [ id for id in current_concept_set[2] if id in filtered_planes_dicts_dict.keys()]
                        old_llc_ids_dict = [filtered_planes_dicts_dict[old_llc_id] for old_llc_id in old_llc_ids]

                        if len(set(old_llc_ids)) > 1:
                            concept_dict = {}
                            node_id_offsets_per_concept = {"room": 1000, "wall": 2000}
                            concept_dict["id"] = hlc_id + node_id_offsets_per_concept[inferred_concept]
                            concept_dict["ws_ids"] = old_llc_ids
                            concept_dict["ws_xy_types"] = [old_llc_id_dict["xy_type"] for old_llc_id_dict in old_llc_ids_dict]
                            concept_dict["ws_msgs"] = [old_llc_id_dict["msg"] for old_llc_id_dict in old_llc_ids_dict]
                            concept_dict["old_llc_ids_dict"] = old_llc_ids_dict
                            concept_dict["center"], mc_entropy, cov_matrices, graph_to_sgraphs = self.add_hlc_node(graph_to_sgraphs, old_llc_ids, concept_dict["id"], inferred_concept)
                            # times["met_gnn_inference"] = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]
                            lo, hi = 0., 2.5
                            mc_entropy_norm  = min(1.0, max(0.0, (abs(mc_entropy) - lo) / (hi - lo)))
                            mc_confidence_norm = 1 - mc_entropy_norm
                            
                            if not "covariance" in self.ablations:
                                semantic_weight_ratio = 1.0
                                for ablation in self.ablations:
                                    splits = ablation.split("_")
                                    if splits[0] == "swr":
                                        if splits[1] == "min":
                                            semantic_weight_ratio = "min"
                                            metric_confidence = mc_confidence_norm
                                            combined_confidence = min(semantic_confidence, mc_confidence_norm)

                                        elif splits[1] == "mean":
                                            semantic_weight_ratio = "mean"
                                            metric_confidence = mc_confidence_norm
                                            combined_confidence = (semantic_confidence + metric_confidence) / 2.0

                                        elif splits[1] == "bayes":
                                            semantic_weight_ratio = "bayes"
                                            epsilon = 0.0001
                                            semantic_variance2 = (1 - semantic_confidence + epsilon) / (semantic_confidence + epsilon)
                                            metric_variance2 = mc_entropy
                                            metric_confidence = metric_variance2
                                            combined_variance2 = (semantic_variance2 * metric_variance2) / (semantic_variance2 + metric_variance2)
                                            combined_confidence = 1 / (1 + combined_variance2)

                                        elif splits[1] == "HW": # Hard Weighting
                                            semantic_weight_ratio = float(splits[2])
                                            metric_confidence = mc_confidence_norm
                                            semantic_weight, metric_weight = semantic_weight_ratio, 1 - semantic_weight_ratio
                                            combined_confidence = semantic_weight * semantic_confidence + metric_weight * metric_confidence
                                            self.get_logger().info(f"dbg semantic_weight_ratio {semantic_weight_ratio} semantic_weight {semantic_weight} metric_weight {metric_weight}")
                                            lin_cov = 1 - combined_confidence
                                            a, b, k = 0.0001, 10, 1.5
                                            exp_cov = a * (b / a) ** (lin_cov ** k)
                                            concept_dict["covariance"] = exp_cov
                                            concept_dict["covariance_lin"] = lin_cov
                                            full_cov = np.zeros((6, 6))
                                            full_cov[0, 0] = exp_cov
                                            full_cov[1, 1] = exp_cov
                                            concept_dict["full_cov"] = full_cov

                                        elif splits[1] == "WC":
                                            # self.get_logger().info(f"dbg semantic_confidence {semantic_confidence}")
                                            # self.get_logger().info(f"dbg cov_matrices {cov_matrices}")
                                            semantic_confidence = max(semantic_confidence, 1e-2) 
                                            if semantic_confidence > 0:
                                                scale = float(splits[2])
                                                scaled_cov = cov_matrices / (semantic_confidence * scale) 
                                            else:
                                                scaled_cov = cov_matrices * 1e6
                                            # self.get_logger().info(f"dbg scale {scale}")
                                            # self.get_logger().info(f"dbg scaled_cov {scaled_cov}")
                                            # Embed into full 6x6 covariance matrix
                                            full_cov = np.zeros((6, 6))
                                            full_cov[0:2, 0:2] = scaled_cov
                                            # self.get_logger().info(f"dbg full_cov {full_cov}")

                                            # (Optional) Set very high uncertainty for unknown orientation
                                            # full_cov[3:, 3:] = np.eye(3) * 99999.0

                                            concept_dict["full_cov"] = full_cov
                                            combined_confidence = 0. ### TODO remove all that
                                            metric_confidence = 0. ### TODO remove all that

                                        else:
                                            semantic_weight_ratio = float(splits[1])
                                            metric_confidence = mc_confidence_norm
                                            semantic_weight, metric_weight = semantic_weight_ratio, 1 - semantic_weight_ratio
                                            combined_confidence = semantic_weight * semantic_confidence + metric_weight * metric_confidence

                                            lin_cov = 1 - combined_confidence
                                            a, b, k = 0.0001, 10, 1.5
                                            exp_cov = a * (b / a) ** (lin_cov ** k)
                                            concept_dict["covariance"] = exp_cov
                                            concept_dict["covariance_lin"] = lin_cov

                                            full_cov = np.zeros((6, 6))
                                            full_cov[0, 0] = exp_cov
                                            full_cov[1, 1] = exp_cov
                                            concept_dict["full_cov"] = full_cov

                                lin_cov = 1 - combined_confidence
                                a, b, k = 0.0001, 10, 1.5
                                exp_cov = a * (b / a) ** (lin_cov ** k)
                                concept_dict["covariance"] = exp_cov
                                concept_dict["covariance_lin"] = lin_cov
                            else:
                                concept_dict["covariance"] = 0.00011
                                concept_dict["covariance_lin"] = 0.00011
                                full_cov = np.zeros((6, 6))
                                full_cov[0, 0] = concept_dict["covariance"]
                                full_cov[1, 1] = concept_dict["covariance"]
                                concept_dict["full_cov"] = full_cov
                            mapped_inferred_concept.append(concept_dict)

                mapped_inferred_concepts[inferred_concept] = mapped_inferred_concept

            times["final"] = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]
            # fig = visualize_nxgraph(graph_to_sgraphs, image_name = f"graph_to_sgraphs", include_node_ids= True, visualize_alone=False)
            # fig.savefig(self.generation_plots_path + f"/graph_to_sgraphs_{self.generation_i}.png")

            if "publications" not in self.ablations:
                if mapped_inferred_concepts and target_concept == "room":
                    self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(mapped_inferred_concepts))

                elif mapped_inferred_concepts and target_concept == "wall":
                    self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(mapped_inferred_concepts))

                elif target_concept == "RoomWall":
                    if mapped_inferred_concepts["room"]:
                        self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(mapped_inferred_concepts["room"]))

                    if mapped_inferred_concepts["wall"]:
                        self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(mapped_inferred_concepts["wall"]))

                    self.v_sgraphs_markers_publisher.publish(self.generate_v_sgraphs_markers_msg(mapped_inferred_concepts))
                        
            elapsed_time_ms = self.get_clock().now().nanoseconds // 1_000_000 - times["start"]
            # elapsed_time_ms = elapsed_time.nanoseconds / 1_000_000         # convert to ms

            self.order_time_log_writer.writerow({"order": prox_graph_order, "times": times, "n_found_concepts": n_found_concepts})
            self.order_time_log_file.flush()

            self.get_logger().info(f'infer_from_planes: process_cb took {elapsed_time_ms:.1f} ms')

            ### Create Rooms to Sgraph graph
            markersize_augment = 3
            graph_to_sgraphs_rooms = copy.deepcopy(graph_to_sgraphs)
            viz_values = {}
            markersize_values = {}
            for i, concept_dict in enumerate(mapped_inferred_concepts["room"]):
                for node_id in concept_dict["ws_ids"]:
                    viz_values.update({node_id: self.colors[concept_dict["id"]%len(self.colors)]})
                markersize_values.update({concept_dict["id"]: concept_dict["covariance_lin"] * markersize_augment}) 
            graph_to_sgraphs_rooms.set_node_attributes("viz_feat", viz_values)
            graph_to_sgraphs_rooms.set_node_attributes("markersize", markersize_values)
            graph_to_sgraphs_rooms = graph_to_sgraphs_rooms.filter_graph_by_node_types(["room", "ws"])
            fig = visualize_nxgraph(graph_to_sgraphs_rooms, image_name = f"inference rooms to sgraph", include_node_ids= False, visualize_alone=False, logger = self.get_logger())
            self.gnns[target_concept].graphs_subplot.update_plot_with_figure(f"Rooms to Sgraph", fig, square_it = True)
            plt.close(fig)


            ### Create Walls to Sgraph graph
            graph_to_sgraphs_walls = copy.deepcopy(graph_to_sgraphs)
            viz_values = {}
            markersize_values = {}
            for i, concept_dict in enumerate(mapped_inferred_concepts["wall"]):
                for node_id in concept_dict["ws_ids"]:
                    viz_values.update({node_id: self.colors[concept_dict["id"]%len(self.colors)]})
                markersize_values.update({concept_dict["id"]: concept_dict["covariance_lin"] * markersize_augment}) 
            graph_to_sgraphs_walls.set_node_attributes("viz_feat", viz_values)
            graph_to_sgraphs_walls.set_node_attributes("markersize", markersize_values)
            graph_to_sgraphs_walls = graph_to_sgraphs_walls.filter_graph_by_node_types(["wall", "ws"])
            fig = visualize_nxgraph(graph_to_sgraphs_walls, image_name = f"inference wall to sgraph", include_node_ids= True, visualize_alone=False)
            self.gnns[target_concept].graphs_subplot.update_plot_with_figure(f"Walls to Sgraph", fig, square_it = True)
            plt.close(fig)
            self.gnns[target_concept].graphs_subplot.save(self.generation_plots_path + f"/HLC_to_sgraph_{self.generation_i}.png")

            self.video_updater.update_figure(self.gnns[target_concept].graphs_subplot.fig)

            self.generation_i += 1

        else:
            self.get_logger().info(f"Graph Reasoning: No edges in the graph!!!")


    # def infer_from_rooms(self, target_concept, msg):
    #     if self.tmp_room_history:
    #         graph = GraphWrapper()
            # for i, room_center in enumerate(self.tmp_room_history):
            #     graph.add_nodes([(i,{"type" : "room","center" : room_center, "x" : room_center,\
            #                         "viz_type" : "Point", "viz_data" : room_center, "viz_feat" : 'ro'})])

            # inferred_concepts = self.gnns[target_concept].cluster_floors(graph)

            # self.get_logger().info(f"flag inferred_concepts {inferred_concepts}")

    def generate_room_subgraph_msg(self, inferred_rooms):
        rooms_msg = RoomsDataMsg()
        for room in inferred_rooms:
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
                room_msg.id = room["id"]
                room_msg.planes = room["ws_msgs"]
                # room_msg.room_center.pose = PoseMsg()
                room_msg.room_center.pose.position.x = float(room["center"][0])
                room_msg.room_center.pose.position.y = float(room["center"][1])
                room_msg.room_center.pose.position.z = float(room["center"][2])
                
                if "full_cov" in room.keys():
                    # self.get_logger().info(f"dbg full_cov.flatten().tolist() {room['full_cov'].flatten().tolist()}")
                    room_msg.room_center.covariance = room["full_cov"].flatten().tolist()
                
                elif "covariance" in room.keys():
                    self.get_logger().info(f"dbg WRONG FLAG: USING OLD COVARIANCE DEFINITION")
                    room_msg.room_center.covariance[0] = room["covariance"]
                    room_msg.room_center.covariance[6] = room["covariance"]

                rooms_msg.rooms.append(room_msg)

        return rooms_msg
    
    def generate_v_sgraphs_markers_msg(self, inferred_concepts):
        FRAME_ID = "map"
        CUBE_SIZE = 0.5
        LINE_SIZE = 0.05
        now = self.get_clock().now().to_msg()
        ma  = MarkerArrayMsg()
        room_height = 9.0
        plane_height = 5.0

        index = 1000
        for concept_name in inferred_concepts.keys():
            for idx, room in enumerate(inferred_concepts[concept_name]):
                c = np.asarray(room["center"], dtype=float)
                room_id = int(room["id"])

                if concept_name == "room":
                    color = ColorRGBSMsg(r=1.0, g=0.0, b=0.0, a=1.0)
                elif concept_name == "wall":
                    color = ColorRGBSMsg(r=0.6, g=0.3, b=0.0, a=1.0)
                lifetime = DurationMsg(sec=10)
                m = MarkerMsg(
                    header=HeaderMsg(stamp=now, frame_id=FRAME_ID),
                    id=index,
                    type=MarkerMsg.CUBE,
                    action=MarkerMsg.ADD,
                    pose=PoseMsg(
                        position=PointMsg(x=c[0], y=c[1], z=room_height),
                        orientation=QuaternionMsg(w=1.0),
                    ),
                    scale=Vector3Msg(x=CUBE_SIZE, y=CUBE_SIZE, z=CUBE_SIZE),
                    color=color,
                    lifetime=lifetime,     # 0 → forever
                )
                index += 1
                ma.markers.append(m)
                # Marker for lines to planes (LINE_LIST)
                line_marker = MarkerMsg(
                    header=HeaderMsg(stamp=now, frame_id=FRAME_ID),
                    ns="room_to_planes",
                    id=index,  # offset ID to avoid conflict
                    type=MarkerMsg.LINE_LIST,
                    action=MarkerMsg.ADD,
                    scale=Vector3Msg(x=LINE_SIZE, y=LINE_SIZE, z=LINE_SIZE),  # only x matters for LINE_LIST
                    color=ColorRGBSMsg(r=0.5, g=0.5, b=0.5, a=1.0),
                    lifetime=lifetime,
                    frame_locked=False,
                    points=[],
                )
                index += 1
                room_point = PointMsg(x=c[0], y=c[1], z=room_height)

                for entry in room.get("old_llc_ids_dict", []):
                    plane_center = entry["center"]
                    plane_point = PointMsg(x=plane_center[0], y=plane_center[1], z=plane_height)

                    line_marker.points.append(room_point)
                    line_marker.points.append(plane_point)

                ma.markers.append(line_marker)

        return ma
    
    def remove_room_from_sgraphs(self, room_id):
        request = RemoveRoomSrv.Request()
        request.room_id = room_id
        self.remove_room_client.call_async(request)

    def add_hlc_node(self, graph, community, hlc_id, hlc_concept):
        if hlc_concept == "room":
            if "naive_factors" in self.ablations:
                factor_name = "room_naive"
                compute_mc_entropy = False
            else:
                factor_name = "room_msd"
                compute_mc_entropy = True
        elif hlc_concept == "wall":    
            factor_name = "wall_naive"
            compute_mc_entropy = False

        cov_matrices = np.zeros((2, 2))

        if self.use_gnn_factors:
            max_d = 1.
            planes_centers_normalized = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, 1]) for node_id in community])
            planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]]) for node_id in community]
            planes_feats_4p = np.array([self.correct_plane_direction_ndarray(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p])
            planes_feats_4p = torch.tensor(planes_feats_4p, dtype=torch.float32) if isinstance(planes_feats_4p, np.ndarray) else planes_feats_4p
            infinite_planes_cp = planes_feats_4p[:, :2] * planes_feats_4p[:, 3:].view(-1, 1)
            # x = torch.cat((torch.tensor(planes_centers_normalized, dtype=torch.float32), 
            #             planes_feats_4p[:, :3].float()), dim=1)
            if "finite_planes" not in self.ablations:
                x_tmp = infinite_planes_cp
            else:
                x_tmp = torch.cat((torch.tensor(planes_centers_normalized, dtype=torch.float32), 
                                   planes_feats_4p[:, :3].float()), dim=1)
            x = x_tmp
            zeros_row = torch.zeros(1, x.size(1), dtype=torch.float32)  # REMOVE THIS FROM F-GNN architecture
            x = torch.cat((x, zeros_row), dim=0)
            x1, x2 = [], []
            for i in range(x.size(0) - 1):
                x1.append(i)
                x2.append(x.size(0) - 1)
            edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
            batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
            if not compute_mc_entropy:
                nn_outputs = self.factor_nn_bridges.infer(x, edge_index, batch, factor_name).numpy()[0]
                mc_entropy = 0.0
            else:
                nn_outputs, mc_entropy, cov_matrices = self.factor_nn_objects.inference(x, edge_index, batch, use_mc_dropout = True)
                nn_outputs, mc_entropy, cov_matrices = nn_outputs.numpy()[0], abs(mc_entropy.numpy()[0]), cov_matrices[0]

            center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
        else:
            center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in community]).astype(np.float32), axis = 0)/len(community)
            mc_entropy = 0.0
         
        node_viz_feat_per_concept = {"room": 'ro', "wall": 'mo'}
        edge_viz_feat_per_concept = {"room": 'red', "wall": 'brown'}

        graph.add_nodes([(hlc_id,{"type" : hlc_concept, "center" : center[:2], "viz":{"type" : "Point","center" : center, "feat" : node_viz_feat_per_concept[hlc_concept]}, "mc_entropy":mc_entropy, "cov_matrices": cov_matrices})])
        
        for node_id in list(set(community)):
            graph.add_edges([(hlc_id, node_id, {"type": f"ws_belongs_{hlc_concept}", "x": [], "viz_feat" : edge_viz_feat_per_concept[hlc_concept], "linewidth":1.0, "alpha":0.5})])

        return center, mc_entropy, cov_matrices, graph
        

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
    
    def split_to_vector3_lists_by_color(self, cloud_msg):
        colour_buckets: Dict[Tuple[int, int, int], List[Vector3Msg]] = defaultdict(list)

        # read_points respects the layout in cloud_msg.fields, so offsets are always right
        for x, y, z, rgb_f in pc2.read_points(cloud_msg,
                                            field_names=('x', 'y', 'z', 'rgb'),
                                            skip_nans=True):
            # reinterpret packed float32 → uint32, little-endian
            rgb_i = struct.unpack('<I', struct.pack('<f', rgb_f))[0]

            r = (rgb_i >> 16) & 0xFF
            g = (rgb_i >> 8)  & 0xFF
            b =  rgb_i        & 0xFF
            key = (r, g, b)

            colour_buckets[key].append(Vector3Msg(x=x, y=y, z=z))

        return colour_buckets
    

    def to_open3d(self, colour_buckets,
              axis_len: float = 0.5,       # physical length of X/Y/Z axes (metres)
              include_axes: bool = True):
        pts, cols = [], []
        for (r, g, b), vectors in colour_buckets.items():
            pts.extend((v.x, v.y, v.z) for v in vectors)
            cols.extend((r / 255, g / 255, b / 255) for _ in vectors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))

        geometries = [pcd]

        if include_axes:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_len, origin=[0.0, 0.0, 0.0]
            )
            geometries.append(frame)

        o3d.visualization.draw_geometries(geometries)
        return pcd
    
    def parse_arguments(self, args):
        parser = argparse.ArgumentParser(description='Process some strings.')
        parser.add_argument('--generated_entities', type=str, default='[]',
                            help='A list of strings')
        parser.add_argument('--use_gnn_factors', type=bool, default=True,
                            help='Use a GNNs as the factors')
        parser.add_argument('--log_path', type=str, default='.',
                            help='Experiment log path')
        parser.add_argument('--ablations', type=str, default='.',
                            help='Experiment log path')
        args, unknown = parser.parse_known_args()

        args.generated_entities = ast.literal_eval(args.generated_entities)
        return args



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

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 4
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=0.0
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=0.0
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 5
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=d
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=d
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=-1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 6
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=0.0
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=0.0 + d + w
        # plane_msg.plane_points[1].y=d
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 7
        # plane_msg.plane_points[0].x=d + d + w
        # plane_msg.plane_points[0].y=0.0
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=d
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=-1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 8
        # plane_msg.plane_points[0].x=0.0
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d
        # plane_msg.plane_points[1].y=0.0 + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 9
        # plane_msg.plane_points[0].x=0.0
        # plane_msg.plane_points[0].y=d + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=-1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 10
        # plane_msg.plane_points[0].x=0.0
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=0.0
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 11
        # plane_msg.plane_points[0].x=d
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=-1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 12
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=0.0 + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 13
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=d + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=0.0
        # plane_msg.ny=-1.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 14
        # plane_msg.plane_points[0].x=0.0 + d + w
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=0.0 + d + w
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        # plane_msg = PlaneMsgFake()
        # plane_msg.id = 15
        # plane_msg.plane_points[0].x=d + d + w
        # plane_msg.plane_points[0].y=0.0 + d + w
        # plane_msg.plane_points[0].z=0.0

        # plane_msg.plane_points[1].x=d + d + w
        # plane_msg.plane_points[1].y=d + d + w
        # plane_msg.plane_points[1].z=0.0

        # plane_msg.nx=-1.0
        # plane_msg.ny=0.0
        # plane_msg.nz=0.0

        # plane_msgs.append(plane_msg)

        return plane_msgs

def main(args=None):
    rclpy.init(args=args)

    graph_reasoning_node = GraphReasoningNode(args)

    # rclpy.spin(graph_reasoning_node)
    # rclpy.get_logger().warn('Destroying node!')
    # graph_reasoning_node.video_updater.stop()
    # graph_reasoning_node.destroy_node()
    # rclpy.shutdown()

    try:
        rclpy.spin(graph_reasoning_node)
    except KeyboardInterrupt:
        graph_reasoning_node.get_logger().warn('KeyboardInterrupt received. Shutting down...')
    except Exception as e:
        print(f"An error occurred while terminating the process group: {e}")
    finally:
        graph_reasoning_node.get_logger().warn('Destroying node!')
        graph_reasoning_node.video_updater.stop()
        graph_reasoning_node.destroy_node()
        rclpy.shutdown()

    


if __name__ == '__main__':
    main()
