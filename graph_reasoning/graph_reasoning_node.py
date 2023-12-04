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
import ament_index_python
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
from s_graphs.msg import WallsData as WallsDataMsg
from s_graphs.msg import WallData as WallDataMsg

from .GNNWrapper import GNNWrapper
from graph_wrapper.GraphWrapper import GraphWrapper
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_matching.utils import segments_distance, segment_intersection

graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph

class GraphReasoningNode(Node):
    def __init__(self):
        super().__init__('graph_matching')

        self.find_rooms, self.find_walls = True, False
        self.reasoning_package_path = ament_index_python.get_package_share_directory("graph_reasoning")
        with open(os.path.join(self.reasoning_package_path, "config/same_room_training.json")) as f:
            self.graph_reasoning_rooms_settings = json.load(f)
        with open(os.path.join(self.reasoning_package_path, "config/same_wall_training.json")) as f:
            self.graph_reasoning_walls_settings = json.load(f)
        datasets_package_path = ament_index_python.get_package_share_directory("graph_datasets")
        with open(os.path.join(datasets_package_path, "config/graph_reasoning.json")) as f:
            dataset_settings = json.load(f)
        dataset_settings["training_split"]["val"] = 0.0
        dataset_settings["training_split"]["test"] = 0.0
        
        self.dataset_settings = dataset_settings
        self.elapsed_times = []
        self.prepare_report_folder()
        
        self.gnns = {"room": GNNWrapper(self.graph_reasoning_rooms_settings, self.report_path, self.get_logger()),
                    "wall": GNNWrapper(self.graph_reasoning_walls_settings, self.report_path, self.get_logger())}
        
        if self.find_rooms:
            self.gnns["room"].define_GCN()
            self.gnns["room"].pth_path = os.path.join(self.reasoning_package_path, "pths/model_rooms.pth")
            self.gnns["room"].load_model()
            self.gnns["room"].save_model(os.path.join(self.report_path,"model_room.pth"))
        if self.find_walls:
            self.gnns["wall"].define_GCN()
            self.gnns["wall"].pth_path = os.path.join(self.reasoning_package_path, "pths/model_walls.pth")
            self.gnns["wall"].load_model() 
            self.gnns["wall"].save_model(os.path.join(self.report_path,"model_wall.pth")) 

        self.synthetic_dataset_generator = SyntheticDatasetGenerator(dataset_settings, self.get_logger(), self.report_path)
        self.set_interface()
        self.get_logger().info(f"Graph Reasoning: Initialized")
        self.node_start_time = time.perf_counter()
        self.first_room_detected = False
  

    def prepare_report_folder(self):
        self.report_path = os.path.join(self.reasoning_package_path, "reports/ros_node/tmp")
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
        combined_settings = {"dataset": self.dataset_settings, "graph_reasoning_rooms": self.graph_reasoning_rooms_settings,\
                             "graph_reasoning_walls": self.graph_reasoning_walls_settings}
        with open(os.path.join(self.report_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)

        
    def set_interface(self):
        if self.find_walls:
            self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/all_map_planes', self.s_graph_all_planes_callback, 10)
        if self.find_rooms:
            self.s_graph_subscription = self.create_subscription(PlanesDataMsg,'/s_graphs/map_planes', self.s_graph_last_planes_callback, 10)

        self.room_subgraph_publisher = self.create_publisher(RoomsDataMsg, '/room_segmentation/room_data', 10)
        self.wall_subgraph_publisher = self.create_publisher(WallsDataMsg, '/wall_segmentation/wall_data', 10)

    def s_graph_all_planes_callback(self, msg):
        # self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in ALL planes topic")
        self.infer("wall", msg)

    def s_graph_last_planes_callback(self, msg):
        # self.get_logger().info(f"Graph Reasoning: {len(msg.x_planes)} X and {len(msg.y_planes)} Y planes received in LAST planes topic")
        self.infer("room", msg)

    def infer(self, target_concept, msg):
        if len(msg.x_planes) == 0 or len(msg.y_planes) == 0:
            return
        
        graph = GraphWrapper()
        if target_concept == "wall":
            target_relation = "ws_same_wall"
        elif target_concept == "room":
            target_relation = "ws_same_room"

        # preprocess features and create graph
        
        planes_msgs = msg.x_planes + msg.y_planes
        planes_dict = []
        # self.get_logger().info(f"Graph Reasoning: characterizing wall surfaces for {target_concept}")
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

        visualize_nxgraph(graph, image_name = f"received - {target_concept}")


        # Inference
        extended_dataset = self.synthetic_dataset_generator.extend_nxdataset([graph], target_relation, "final")
        if len(extended_dataset["train"][0].get_edges_ids()) > 0:
            extended_dataset.pop("test"), extended_dataset.pop("val")
            normalized_dataset = self.synthetic_dataset_generator.normalize_features_nxdatset(extended_dataset)
            # self.get_logger().info(f"Graph Reasoning: Inferring")
            # start_time = time.perf_counter()
            inferred_concepts = self.gnns[target_concept].infer(normalized_dataset["train"][0],True)
            # for inferred_concept in inferred_concepts:
            #     self.get_logger().info(f"flag inferred_concept {inferred_concept}")
            # end_time = time.perf_counter()
            # self.elapsed_times.append(end_time - start_time)
            # self.avg_elapsed_time = np.average(self.elapsed_times)
            # f = open(f"/.../computing_time_{target_concept}.txt","w+")
            # for i in range(10):
            #     f.write(f"computed time {self.avg_elapsed_time} \n")
            # f.close()
            # self.get_logger().info(f"Graph Reasoning: average elapsed time {self.avg_elapsed_time}secs")

            # Prepare message to SGraphs
            mapped_inferred_concepts = []
            for concept in inferred_concepts:
                concept_dict = copy.deepcopy(concept)
                concept_dict["ws_ids"] = [splitting_mapping[ws_id]["old_id"] for ws_id in concept["ws_ids"]]
                concept_dict["ws_xy_types"] = [splitting_mapping[ws_id]["xy_type"] for ws_id in concept["ws_ids"]]
                concept_dict["ws_msgs"] = [splitting_mapping[ws_id]["msg"] for ws_id in concept["ws_ids"]]
                mapped_inferred_concepts.append(concept_dict)

            if mapped_inferred_concepts and target_concept == "room":
                self.room_subgraph_publisher.publish(self.generate_room_subgraph_msg(mapped_inferred_concepts))
                # self.get_logger().info(f"Graph Reasoning: published {len(inferred_concepts)} rooms")
            elif mapped_inferred_concepts and target_concept == "wall":
                self.wall_subgraph_publisher.publish(self.generate_wall_subgraph_msg(mapped_inferred_concepts))
                # self.get_logger().info(f"Graph Reasoning: published {len(inferred_concepts)} walls")

        
        else:
            self.get_logger().info(f"Graph Reasoning: No edges in the graph!!!")
        

    def generate_room_subgraph_msg(self, inferred_rooms):
        rooms_msg = RoomsDataMsg()
        for room_id, room in enumerate(inferred_rooms):
            x_planes, y_planes = [], []
            x_centers, y_centers = [], []
            for plane_index, ws_type in enumerate(room["ws_xy_types"]):
                if ws_type == "x":
                    x_planes.append(room["ws_msgs"][plane_index])
                    x_centers.append(room["ws_centers"][plane_index])
                elif ws_type == "y":
                    y_planes.append(room["ws_msgs"][plane_index])
                    y_centers.append(room["ws_centers"][plane_index])

            if len(x_planes) == 2 and len(y_planes) == 2:
                pass
                # if not self.first_room_detected:
                #     elapsed_time = time.perf_counter() - self.node_start_time
                #     f = open(f"/.../FRD_time.txt","w+")
                #     f.write(f"computed time {elapsed_time} \n")
                #     f.close()
                #     self.first_room_detected = True
                #     self.get_logger().info(f"Time of first 4-ws room detection: {elapsed_time}")

            elif len(x_planes) == 1 and len(y_planes) == 2:
                x_planes = []
                room["center"] = (y_centers[0] + y_centers[1])/2


            elif len(x_planes) == 2 and len(y_planes) == 1:
                y_planes = []
                room["center"] = (x_centers[0] + x_centers[1])/2

            else:
                x_planes, y_planes = [], []

            if x_planes or y_planes:

                room_msg = RoomDataMsg()
                room_msg.id = room_id
                room_msg.x_planes = x_planes
                room_msg.y_planes = y_planes
                room_msg.room_center = PoseMsg()
                room_msg.room_center.position.x = float(room["center"][0])
                room_msg.room_center.position.y = float(room["center"][1])
                room_msg.room_center.position.z = float(room["center"][2])
                room_msg.cluster_center.x = float(room["center"][0])
                room_msg.cluster_center.y = float(room["center"][1])
                room_msg.cluster_center.z = float(room["center"][2])
                rooms_msg.rooms.append(room_msg)

        return rooms_msg
    
    def generate_wall_subgraph_msg(self, inferred_walls):
        walls_msg = WallsDataMsg()
        for wall_id, wall in enumerate(inferred_walls):
            x_planes, y_planes = [], []
            x_centers, y_centers = [], []
            for plane_index, ws_type in enumerate(wall["ws_xy_types"]):
                if ws_type == "x":
                    x_planes.append(wall["ws_msgs"][plane_index])
                    x_centers.append(wall["ws_centers"][plane_index])
                elif ws_type == "y":
                    y_planes.append(wall["ws_msgs"][plane_index])
                    y_centers.append(wall["ws_centers"][plane_index])

            if len(x_planes) == 0 and len(y_planes) == 2:
                x_planes = []
                wall["center"] = (y_centers[0] + y_centers[1])/2

            elif len(x_planes) == 2 and len(y_planes) == 0:
                y_planes = []
                wall["center"] = (x_centers[0] + x_centers[1])/2
            
            else:
                x_planes, y_planes = [], []

            if x_planes or y_planes:
                wall_msg = WallDataMsg()
                wall_msg.id = wall_id
                wall_msg.x_planes = x_planes
                wall_msg.y_planes = y_planes
                wall_msg.wall_center = PoseMsg()
                wall_msg.wall_center.position.x = float(wall["center"][0])
                wall_msg.wall_center.position.y = float(wall["center"][1])
                wall_msg.wall_center.position.z = float(wall["center"][2])

                walls_msg.walls.append(wall_msg)

        return walls_msg
    

    # def preprocess_planes_to_graph(self, msg):
    #     graph = GraphWrapper()
    #     if target_concept == "wall":
    #         target_relation = "ws_same_wall"
    #     elif target_concept == "room":
    #         target_relation = "ws_same_room"

    #     # preprocess features and create graph
    #     planes_msgs = msg.x_planes + msg.y_planes
    #     planes_dict = []
    #     self.get_logger().info(f"Graph Reasoning: characterizing wall surfaces for {target_concept}")
    #     for i, plane_msg in enumerate(planes_msgs):
    #         plane_dict = {"id": plane_msg.id, "normal" : np.array([plane_msg.nx,plane_msg.ny,plane_msg.nz])}
    #         plane_dict["xy_type"] = "x" if i<len(msg.x_planes) else "y" 
    #         plane_dict["msg"] = plane_msg
    #         plane_dict["center"], plane_dict["segment"], plane_dict["length"] = self.characterize_ws(plane_msg.plane_points)
    #         planes_dict.append(plane_dict)

    #     filtered_planes_dict = self.filter_overlapped_ws(planes_dict)
    #     splitted_planes_dict = self.split_ws(filtered_planes_dict)
    #     splitting_mapping = {}
    #     for plane_dict in splitted_planes_dict:
    #         def add_ws_node_features(feature_keys, feats):
    #             if feature_keys[0] == "centroid":
    #                 feats = np.concatenate([feats, plane_dict["center"][:2]]).astype(np.float32)
    #             elif feature_keys[0] == "length":
    #                 feats = np.concatenate([feats, [plane_dict["length"]]]).astype(np.float32)   #, [np.log(ws_length)]]).astype(np.float32)
    #             elif feature_keys[0] == "normals":
    #                 feats = np.concatenate([feats, plane_dict["normal"][:2]]).astype(np.float32)
    #             if len(feature_keys) > 1:
    #                 feats = add_ws_node_features(feature_keys[1:], feats)
    #             return feats
    #         x = add_ws_node_features(self.dataset_settings["initial_features"]["ws_node"], [])

    #         graph.add_nodes([(plane_dict["id"],{"type" : "ws","center" : plane_dict["center"], "x" : x, "label": 1, "normal" : plane_dict["normal"],\
    #                                        "viz_type" : "Line", "viz_data" : plane_dict["segment"], "viz_feat" : "black",\
    #                                        "linewidth": 2.0, "limits": plane_dict["segment"]})])
    #         splitting_mapping[plane_dict["id"]] = {"old_id" : plane_dict["old_id"], "xy_type" : plane_dict["xy_type"], "msg" : plane_dict["msg"]}



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
        extension = 0.3
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
    

def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphReasoningNode()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
