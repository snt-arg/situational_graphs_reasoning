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
from shapely.geometry import Polygon

from s_graphs.msg import PlanesData as PlanesDataMsg
from s_graphs.msg import RoomsData as RoomsDataMsg
from s_graphs.msg import RoomData as RoomDataMsg

from .GNNWrapper import GNNWrapper
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_matching.utils import segments_distance, segment_intersection

class GraphReasoningNode(Node):
    def __init__(self):
        super().__init__('graph_matching')

        with open("/home/adminpc/reasoning_ws/src/graph_reasoning/config/same_room_training.json") as f:
            self.graph_reasoning_settings = json.load(f)
        with open("/home/adminpc/reasoning_ws/src/graph_datasets/config/graph_reasoning.json") as f:
            dataset_settings = json.load(f)
        dataset_settings["training_split"]["val"] = 0.0
        dataset_settings["training_split"]["test"] = 0.0
        
        self.dataset_settings = dataset_settings
        self.elapsed_times = []
        self.prepare_report_folder()
        self.gnn = GNNWrapper(self.graph_reasoning_settings, self.report_path, self.get_logger())
        self.gnn.define_GCN()
        self.gnn.pth_path = '/home/adminpc/reasoning_ws/src/graph_reasoning/pths/model.pth'
        self.gnn.load_model() 
        self.gnn.save_model(os.path.join(self.report_path,"model.pth")) 
        self.synthetic_datset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger(), self.report_path)
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")


    # def get_parameters(self):
    #     pass

    def prepare_report_folder(self):
        self.report_path = "/home/adminpc/reasoning_ws/src/graph_reasoning/reports/ros_node/" + self.graph_reasoning_settings["report"]["name"]
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
        self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_planes_callback, 0) # /s_graphs/all_map_planes
        self.room_data_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)

    def s_graph_planes_callback(self, msg):
        self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received")
        graph = GraphWrapper()

        # preprocess features and create graph
        start_time = time.perf_counter()
        planes_msgs = msg.x_planes + msg.y_planes
        planes_dict = []
        self.get_logger().info(f"Graph Reasoning: characterizing wall surfaces")
        for i, plane_msg in enumerate(planes_msgs):
            plane_dict = {"id": plane_msg.id, "normal" : np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])}
            plane_dict["xy_type"] = "x" if i<len(msg.x_planes) else "y" 
            plane_dict["msg"] = plane_msg
            plane_dict["center"], plane_dict["segment"], plane_dict["length"] = self.characterize_ws(plane_msg.plane_points)
            planes_dict.append(plane_dict)

        filtered_planes_dict = self.filter_overlapped_ws(planes_dict)
        splitted_planes_dict = self.split_ws(filtered_planes_dict)
        splitting_mapping = {}
        for plane_dict in splitted_planes_dict:
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
            x = add_ws_node_features(self.dataset_settings["initial_features"]["ws_node"], [])

            graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "x" : x, "label": 1, "normal" : plane_dict["normal"],\
                                           "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
                                           "linewidth": 2.0, "limits": plane_dict["segment"]})])
            splitting_mapping[plane_dict["id"]] = {"old_id" : plane_dict["old_id"], "xy_type" : plane_dict["xy_type"], "msg" : plane_dict["msg"]}

        # Inference
        extended_dataset = self.synthetic_datset_generator.extend_nxdataset([graph], "ws_same_room")
        extended_dataset["test"] = extended_dataset["train"]
        extended_dataset["val"] = extended_dataset["train"]
        normalized_dataset = self.synthetic_datset_generator.normalize_features_nxdatset(extended_dataset)
        self.get_logger().info(f"Graph Reasoning: Inferring")
        inferred_rooms = self.gnn.infer(normalized_dataset["train"][0], True)
        end_time = time.perf_counter()
        self.elapsed_times.append(end_time - start_time)
        avg_elapsed_time = np.average(self.elapsed_times)
        self.get_logger().info(f"Graph Reasoning: average elapsed time {avg_elapsed_time}secs")

        # Prepare message to SGraphs
        mapped_inferred_rooms = []
        for room in inferred_rooms:
            room_dict = copy.deepcopy(room)
            room_dict["ws_ids"] = [splitting_mapping[ws_id]["old_id"] for ws_id in room["ws_ids"]]
            room_dict["ws_xy_types"] = [splitting_mapping[ws_id]["xy_type"] for ws_id in room["ws_ids"]]
            room_dict["ws_msgs"] = [splitting_mapping[ws_id]["msg"] for ws_id in room["ws_ids"]]
            mapped_inferred_rooms.append(room_dict)
        
        if mapped_inferred_rooms:
            self.room_data_publisher.publish(self.generate_room_clustering_msg(mapped_inferred_rooms))
        
        self.get_logger().info(f"Graph Reasoning: published {len(inferred_rooms)} rooms")
        

    def generate_room_clustering_msg(self, inferred_rooms):
        rooms_msg = RoomsDataMsg()
        for room_id, room in enumerate(inferred_rooms):
            x_planes, y_planes = [], []
            for plane_index, ws_type in enumerate(room["ws_xy_types"]):
                if ws_type == "x":
                    x_planes.append(room["ws_msgs"][plane_index])
                elif ws_type == "y":
                    y_planes.append(room["ws_msgs"][plane_index])

            room_msg = RoomDataMsg()
            room_msg.id = room_id
            room_msg.x_planes = x_planes
            room_msg.y_planes = y_planes
            room_msg.room_center = PoseMsg()
            room_msg.room_center.position.x = float(room["center"][0])
            room_msg.room_center.position.y = float(room["center"][1])
            room_msg.room_center.position.z = float(room["center"][2])
            rooms_msg.rooms.append(room_msg)

        return rooms_msg


    def characterize_ws(self, points):
        points = np.array([np.array([point.x,point.y,0]) for point in points])
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
    

    def filter_overlapped_ws(self, planes_dict):
        self.get_logger().info(f"Graph Reasoning: filter overlapped wall surfaces")
        segments = [ plane_dict["segment"] for plane_dict in planes_dict]
        expansion = 0.1
        coverage_thr = 0.8

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

        filteredin_planes_dict = copy.deepcopy(planes_dict)
        filteredout_planes_index = []
        for i, agumented_segment in enumerate(agumented_segments):
            for jj in range(len(agumented_segments) - i - 1):
                j = len(agumented_segments) - jj - 1
                if j not in filteredout_planes_index and (compute_coverage(agumented_segment, agumented_segments[j]) > coverage_thr):
                    filteredin_planes_dict.pop(i)
                    filteredout_planes_index.append(i)
                    break

        return filteredin_planes_dict
            

    def split_ws(self, planes_dict):
        self.get_logger().info(f"Graph Reasoning: splitting wall surfaces")
        extension = 0.3
        thr_length = 0.5
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
    

def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphReasoningNode()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
