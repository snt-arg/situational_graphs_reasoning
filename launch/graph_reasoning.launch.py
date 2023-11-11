import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.events import Shutdown
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # launch_tester_node_ns = LaunchConfiguration('launch_tester_node_ns')

    config = os.path.join(
        get_package_share_directory('graph_reasoning'),
        'config',
        'params.yaml'
    )

    graph_reasoning_node = Node(
        package='graph_reasoning',
        executable='graph_reasoning',
        # namespace='graph_reasoning',
        parameters = [config],
        remappings=[
            ('graph_reasoning/graphs','/s_graphs/graph_structure'),
        ] #TODO change remapping
    )

    return LaunchDescription([
        graph_reasoning_node,
    ])
