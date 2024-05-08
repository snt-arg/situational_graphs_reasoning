#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
    quaternion_from_matrix,
    quaternion_multiply,
)
import numpy as np
import sys


groundtruth_pose_topic_name = "/gazebo/model_states"
# slam_pose_topic_name  =  "/aft_mapped_to_init"
# slam_pose_topic_name  =  "/laser_map"
slam_pose_topic_name = "/s_graphs/odom_pose_corrected"

groundtruth_pose_file = open("stamped_groundtruth.txt", "w+")
groundtruth_pose_file.write("#timestamp  tx ty tz qx qy qz qw\n")
slam_pose_file = open("stamped_traj_estimate.txt", "w+")
print("Created the file, now writing")
slam_pose_file.write("#timestamp  tx ty tz qx qy qz qw\n")
first = False

robot_id = -1
for i in range(1, len(sys.argv)):
    if i == 1:
        robot_id=sys.argv[i]
    elif i == 2:
        groundtruth_pose_topic_name=sys.argv[i]


def groundtruthPoseCallback(groundtruth_pose_msg):
    # print(groundtruth_pose_msg.pose[9].position.x)

    time = rospy.get_rostime().to_sec()
    # print("time : ", time)
    # time = (time.to_sec() + time.to_nsec()) * 1e-9
    tx = groundtruth_pose_msg.pose[int(robot_id)].position.x  # 16
    ty = groundtruth_pose_msg.pose[int(robot_id)].position.y  # 16
    tz = groundtruth_pose_msg.pose[int(robot_id)].position.z  # 16
    rx = groundtruth_pose_msg.pose[int(robot_id)].orientation.x  # 16
    ry = groundtruth_pose_msg.pose[int(robot_id)].orientation.y  # 16
    rz = groundtruth_pose_msg.pose[int(robot_id)].orientation.z  # 16
    rw = groundtruth_pose_msg.pose[int(robot_id)].orientation.w  # 16

    groundtruth_pose_file.write(
        str(time)
        + " "
        + str(tx)
        + " "
        + str(ty)
        + " "
        + str(tz)
        + " "
        + str(rx)
        + " "
        + str(ry)
        + " "
        + str(rz)
        + " "
        + str(rw)
        + "\n"
    )


def slamPoseCallback(slam_pose_msg):
    # print("inside the slam pose callback")
    time = rospy.get_rostime().to_sec()
    # time = (time.to_sec() + time.to_nsec()) * 1e-9
    # time = rospy.get_rostime()
    # time = (time.to_sec() + time.to_nsec()) * 1e-9
    odom_x = slam_pose_msg.pose.position.x
    odom_y = slam_pose_msg.pose.position.y
    odom_z = slam_pose_msg.pose.position.z
    odom_rx = slam_pose_msg.pose.orientation.x
    odom_ry = slam_pose_msg.pose.orientation.y
    odom_rz = slam_pose_msg.pose.orientation.z
    odom_rw = slam_pose_msg.pose.orientation.w

    slam_pose_file.write(
        str(time)
        + " "
        + str(odom_x)
        + " "
        + str(odom_y)
        + " "
        + str(odom_z)
        + " "
        + str(odom_rx)
        + " "
        + str(odom_ry)
        + " "
        + str(odom_rz)
        + " "
        + str(odom_rw)
        + "\n"
    )


def subscribers():

    rospy.init_node("text_file_generator", anonymous=True)
    rospy.Subscriber(groundtruth_pose_topic_name,
                     ModelStates, groundtruthPoseCallback)
    print("should go in")
    rospy.Subscriber(slam_pose_topic_name, PoseStamped, slamPoseCallback)
    rospy.spin()


if __name__ == "__main__":
    subscribers()



### USAGE
# python3 ~/reasoning_ws/src/situational_graphs_reasoning/situational_graphs_reasoning/generate_text_file_s_graphs.py 3 /gazebo/model_states
# evo_ape tum stamped_groundtruth.txt stamped_traj_estimate.txt -va > results.txt
#ros2 service call /s_graphs/save_map s_graphs/srv/SaveMap "{utm: False, resolution: 0.0, destination: '/home/adminpc/Documents/My papers/ICRA 2024/Experiments/Real/construction_site_oetrange/FSC_map.pcd'}"