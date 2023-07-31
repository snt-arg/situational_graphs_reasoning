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

from .GNNWrapper import GNNWrapper
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator

class GraphReasoningNode(Node):
    def __init__(self):
        super().__init__('graph_matching')
        self.get_logger().info(f"Graph Reasoning: Initializing")

        # with open(get_package_share_directory('graph_reasoning'),"config", "same_room_training.json") as f:
        with open("/home/adminpc/reasoning_ws/src/graph_reasoning/config/same_room_training.json") as f:
            self.graph_reasoning_settings = json.load(f)
        with open("/home/adminpc/reasoning_ws/src/graph_datasets/config/graph_reasoning.json") as f:
            dataset_settings = json.load(f)
        
        self.dataset_settings = {}   # TODO Include
        self.prepare_report_folder()
        # self.gm = GNNWrapper(self.graph_reasoning_settings, self.report_path)    
        self.set_interface()
        self.synthetic_datset_generator = SyntheticDatasetGenerator(dataset_settings)


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
        self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_planes_callback, 0)
        # self.room_clustering_srv = self.create_service(PoseMsg, 'graph_matching/subgraph_match', self.room_clustering_srv_callback)


    def s_graph_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received")
        xy_classification = {"x": [plane.id for plane in msg.x_planes], "y": [plane.id for plane in msg.y_planes]}
        graph = GraphWrapper()

        # preprocess features
        planes_msgs = msg.x_planes + msg.y_planes
        for i, plane_msg in enumerate(planes_msgs):
            center, limit_1, limit_2, length = self.caracterize_ws(1 if i < len(msg.x_planes) else 0, plane_msg.plane_points)
            normal = np.array([plane_msg.plane_orientation.x,plane_msg.plane_orientation.y,plane_msg.plane_orientation.z])
            self.get_logger().info(f"Graph Reasoning: normal {normal}")
            x = np.concatenate([center[:2], [length], normal[:2]]).astype(np.float32)
            geometric_info = np.concatenate([center, normal])
            graph.add_nodes([(plane_msg.id,{"type" : "ws","center" : center, "x" : x, "normal" : normal, "Geometric_info" : geometric_info,\
                                           "viz_type" : "Line", "viz_data" : [limit_1,limit_2], "viz_feat" : "black",\
                                           "linewidth": 2.0, "limits": [limit_1,limit_2]})])

        extended_graph = self.synthetic_datset_generator.extend_nxdataset([graph], "ws_same_room")["train"][0]
        visualize_nxgraph(extended_graph, "extended s_graph")

        ### Create graph
        self.raw_ws_graph = None 
        

    def room_clustering_srv_callback(self, request, response):
        self.get_logger().info('Graph Reasoning: Received room clustering request')

        def match_fn(request, response):
            if request.base_graph not in self.gm.graphs.keys() or request.target_graph not in self.gm.graphs.keys() or \
                self.gm.graphs[request.base_graph].is_empty() or self.gm.graphs[request.target_graph].is_empty():
                response.success = 3
            else:
                success, matches = self.gm.match(request.base_graph, request.target_graph)
                
                if success:
                    matches_msg = [self.generate_match_msg(match) for match in matches]
                    matches_visualization_msg = [self.generate_match_visualization_msg(match) for match in matches]
                    self.get_logger().warn('{} successful match(es) found!'.format(len(matches_msg)))
                    response.success = 0 if len(matches_msg) == 1 else 1

                else:
                    response.success = 2
                    self.get_logger().warn('Graph Manager: no good matches found!')

                if response.success == 0:
                    self.unique_match_publisher.publish(matches_msg[0])
                    self.unique_match_visualization_publisher.publish(matches_visualization_msg[0])
                if response.success == 0 or response.success == 1:
                    self.best_match_publisher.publish(matches_msg[0])
                    self.best_match_visualization_publisher.publish(matches_visualization_msg[0])

            return response
        
        response = match_fn(request, response)
        return response


    def generate_room_clustering_msg(self, match):
        msg = MatchMsg()
        for edge in match:
            ### Edge
            edge_msg = EdgeMsg()
            edge_msg.origin_node = edge["origin_node"]
            edge_msg.target_node = edge["target_node"]
            attrib_msg = AttributeMsg()
            attrib_msg.name = "score"
            attrib_msg.fl_value = [edge["score"]]
            edge_msg.attributes = [attrib_msg]
            match_msg.edges.append(edge_msg)
            # graph_msg.name = str(score)

            ### Origin node
            origin_node_msg = NodeMsg()
            origin_node_msg.id = edge["origin_node"]
            origin_node_msg.type = edge["origin_node_attrs"]["type"]
            origin_node_msg.attributes = self.dict_to_attr_msg_list(edge["origin_node_attrs"])
            match_msg.basis_nodes.append(origin_node_msg)


            ### Target node
            target_node_msg = NodeMsg()
            target_node_msg.id = edge["target_node"]
            target_node_msg.type = edge["target_node_attrs"]["type"]
            target_node_msg.attributes = self.dict_to_attr_msg_list(edge["target_node_attrs"])
            match_msg.target_nodes.append(target_node_msg)

        return match_msg


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
