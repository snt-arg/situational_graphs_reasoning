import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.events import Shutdown
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression, TextSubstitution)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # launch_tester_node_ns = LaunchConfiguration('launch_tester_node_ns')

    prefix_command = (
    'bash -c "source /home/adminpc/workspaces/reasoning_ws/src/graph_factor_nn/nn_factors_venv/bin/activate && '
    'echo VIRTUAL_ENV=$VIRTUAL_ENV"'
)
    
    verbosity = DeclareLaunchArgument(
            "log_level",
            default_value= TextSubstitution(text=str("WARN")),
            description="Logging level",
      )

    declare_generated_entities_arg = DeclareLaunchArgument(
        'generated_entities',
        default_value='["default_entity"]',
        description='List of entities to be generated.')

    graph_reasoning_node = Node(
        package='graph_reasoning',
        executable='graph_reasoning',
        # namespace='graph_reasoning',
        arguments=['--generated_entities', LaunchConfiguration('generated_entities'), '--log-level', 'DEBUG'],
        prefix=prefix_command,
        remappings=[
            ('graph_reasoning/graphs','/s_graphs/graph_structure'),
        ] #TODO change remapping
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '-c', f'source /home/adminpc/workspaces/reasoning_ws/src/graph_factor_nn/nn_factors_venv/bin/activate && ros2 run graph_reasoning graph_reasoning_node.py'],
            output='screen'
        ),
    ])
