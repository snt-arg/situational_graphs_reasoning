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
import time, os, json, shutil, sys
import copy
import numpy as np
from rclpy.node import Node
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros.buffer_interface import BufferInterface
import tf2_geometry_msgs 
from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import PointStamped as PointStampedMsg
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Transform as TransformMsg
from geometry_msgs.msg import TransformStamped as TransformStampedMsg
from std_msgs.msg import ColorRGBA as ColorRGBSMsg
from std_msgs.msg import Header as HeaderMsg
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.parameter import Parameter
from rclpy.parameter import ParameterType
from ament_index_python.packages import get_package_share_directory

from s_graphs.msg import PlanesData as PlanesDataMsg
from s_graphs.msg import RoomsData as RoomsDataMsg
from s_graphs.msg import RoomData as RoomDataMsg

from .GNNWrapper import GNNWrapper
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator

class GraphReasoningNode(Node):
    def __init__(self):
        super().__init__('graph_matching')

        # with open(get_package_share_directory('graph_reasoning'),"config", "same_room_training.json") as f:
        with open("/home/adminpc/reasoning_ws/src/graph_reasoning/config/same_room_training.json") as f:
            self.graph_reasoning_settings = json.load(f)
        with open("/home/adminpc/reasoning_ws/src/graph_datasets/config/graph_reasoning.json") as f:
            dataset_settings = json.load(f)
        dataset_settings["training_split"]["val"] = 0.0
        dataset_settings["training_split"]["test"] = 0.0
        
        self.dataset_settings = {}   # TODO Include
        self.prepare_report_folder()
        self.gnn = GNNWrapper(self.graph_reasoning_settings, self.report_path, self.get_logger())
        self.gnn.define_GCN()
        self.gnn.pth_path = '/home/adminpc/reasoning_ws/src/graph_reasoning/pths/model.pth'
        self.gnn.load_model() 
        self.synthetic_datset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger())
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")


    # def get_parameters(self):
    #     pass

    def prepare_report_folder(self):
        self.report_path = "/home/adminpc/reasoning_ws/src/graph_reasoning/reports/" + self.graph_reasoning_settings["report"]["name"] + "_s_graph"
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
        combined_settings = {"dataset": self.dataset_settings, "graph_reasoning": self.graph_reasoning_settings}
        with open(os.path.join(self.report_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)
        
    def set_interface(self):
        self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/map_planes', self.s_graph_planes_callback, 0) # /s_graphs/all_map_planes
        self.room_data_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)

    def s_graph_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received")
        xy_classification = {"x": [plane.id for plane in msg.x_planes], "y": [plane.id for plane in msg.y_planes]}
        len_x_planes = len(msg.x_planes)
        graph = GraphWrapper()

        # preprocess features
        planes_msgs = msg.x_planes + msg.y_planes
        for i, plane_msg in enumerate(planes_msgs):
            center, limit_1, limit_2, length = self.caracterize_ws(1 if i < len(msg.x_planes) else 0, plane_msg.plane_points)
            normal = np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])
            x = np.concatenate([center[:2], [length], normal[:2]]).astype(np.float32)
            # x = np.concatenate([[length], normal[:2]]).astype(np.float32)
            graph.add_nodes([(plane_msg.id,{"type" : "ws","center" : center, "x" : x, "label": 1, "normal" : normal,\
                                           "viz_type" : "Line", "viz_data" : [limit_1,limit_2], "viz_feat" : "black",\
                                           "linewidth": 2.0, "limits": [limit_1,limit_2]})])

        remapping = graph.relabel_nodes() ### TODO check
        extended_dataset = self.synthetic_datset_generator.extend_nxdataset([graph], "ws_same_room")
        extended_dataset["test"] = extended_dataset["train"]
        extended_dataset["val"] = extended_dataset["train"]
        inferred_rooms = self.gnn.infer(extended_dataset["train"][0], True)
        
        if inferred_rooms:
            self.room_data_publisher.publish(self.generate_room_clustering_msg(inferred_rooms, planes_msgs, len_x_planes))

    def generate_room_clustering_msg(self, inferred_rooms, planes_msgs, len_x_planes):
        rooms_msg = RoomsDataMsg()
        for id, room in enumerate(inferred_rooms):
            x_planes, y_planes = [], []
            for plane_id in room["ws_ids"]:
                if plane_id < len_x_planes:
                    x_planes.append(planes_msgs[plane_id])
                elif plane_id >= len_x_planes:
                    y_planes.append(planes_msgs[plane_id])

            room_msg = RoomDataMsg()
            room_msg.id = id
            room_msg.x_planes = x_planes
            room_msg.y_planes = y_planes
            room_msg.room_center = PoseMsg()
            room_msg.room_center.position.x = float(room["center"][0])
            room_msg.room_center.position.y = float(room["center"][1])
            room_msg.room_center.position.z = float(room["center"][2])
            rooms_msg.rooms.append(room_msg)

        return rooms_msg


    def caracterize_ws(self, xy, points):
        points = [[point.x,point.y,point.z] for point in points]
        long_dim = [point[xy] for point in points]
        width_avg = np.average([point[abs(xy-1)] for point in points])

        limit_1 = np.array([0,0,0])
        limit_1[xy] = max(long_dim)
        limit_1[abs(xy-1)] = width_avg

        limit_2 = np.array([0,0,0])
        limit_2[xy] = min(long_dim)
        limit_2[abs(xy-1)] = width_avg

        center = np.array([0,0,0])
        center[xy] = limit_2[xy]/2 + limit_1[xy]/2
        center[abs(xy-1)] = width_avg

        length = np.linalg.norm(limit_1 - limit_2)

        return center, limit_1, limit_2, length


def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphReasoningNode()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
