"""Full cell: camera -> detector -> sorter -> arm.

    ros2 launch intelpick intelpick.launch.py dry_run:=true            # no arm attached
    ros2 launch intelpick intelpick.launch.py serial_port:=/dev/ttyUSB0 model:=m2
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('intelpick')
    args = {
        'params': os.path.join(share, 'config', 'intelpick.yaml'),
        'calibration_file': os.path.expanduser('~/.intelpick/calibration.yaml'),
        'dry_run': 'false',
        'model': 'm2',
        'serial_port': '/dev/ttyUSB0',
        'backend': 'color',
        'sorter': 'true',
    }
    cfg = {k: LaunchConfiguration(k) for k in args}
    params = cfg['params']
    calib = {'calibration_file': cfg['calibration_file']}

    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        Node(package='intelpick', executable='camera_node', name='camera', parameters=[params]),
        Node(package='intelpick', executable='detector_node', name='detector',
             parameters=[params, calib, {'backend': cfg['backend']}]),
        Node(package='intelpick', executable='arm_node', name='arm',
             parameters=[params, {'dry_run': cfg['dry_run'], 'model': cfg['model'],
                                  'serial_port': cfg['serial_port']}]),
        Node(package='intelpick', executable='sorter_node', name='sorter',
             parameters=[params, calib], condition=IfCondition(cfg['sorter'])),
    ])
