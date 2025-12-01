# Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
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

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('depth_anything_v3')
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'depth_anything_v3.param.yaml'),
        description='Path to the parameter file'
    )
    
    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic',
        default_value='/camera/image_raw/compressed',
        description='Input compressed image topic'
    )
    
    input_camera_info_topic_arg = DeclareLaunchArgument(
        'input_camera_info_topic',
        default_value='/camera/camera_info',
        description='Input camera info topic'
    )
    
    output_depth_topic_arg = DeclareLaunchArgument(
        'output_depth_topic',
        default_value='/depth_anything_v3/output/depth_image',
        description='Output depth image topic'
    )
    
    output_point_cloud_topic_arg = DeclareLaunchArgument(
        'output_point_cloud_topic',
        default_value='/depth_anything_v3/output/point_cloud',
        description='Output point cloud topic'
    )

    # Depth Anything V3 node
    depth_anything_v3_node = Node(
        package='depth_anything_v3',
        executable='depth_anything_v3_main',
        name='depth_anything_v3',
        output='screen',
        remappings=[
            ('~/input/image', LaunchConfiguration('input_image_topic')),
            ('~/input/camera_info', LaunchConfiguration('input_camera_info_topic')),
            ('~/output/depth_image', LaunchConfiguration('output_depth_topic')),
            ('~/output/point_cloud', LaunchConfiguration('output_point_cloud_topic'))
        ],
        parameters=[LaunchConfiguration('params_file')]
    )

    return LaunchDescription([
        # Launch arguments
        params_file_arg,
        input_image_topic_arg,
        input_camera_info_topic_arg,
        output_depth_topic_arg,
        output_point_cloud_topic_arg,
        # Nodes
        depth_anything_v3_node,
    ])
