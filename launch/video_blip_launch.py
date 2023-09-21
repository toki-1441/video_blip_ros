import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
  config = os.path.join(
    get_package_share_directory('video_blip_ros'),
    'config',
    'video_blip.param.yaml'
    )
  return LaunchDescription([
    Node(
      package='video_blip_ros',
      executable='video_blip',
      parameters=[config]
    )
  ])