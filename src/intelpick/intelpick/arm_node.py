"""ROS 2 wrapper around RoArm: /arm/move_to, /arm/set_gripper, /arm/home.

Services run on the default single-threaded executor, so arm commands never interleave.
Units are metres on the ROS side and millimetres on the wire.
"""

import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_srvs.srv import Trigger

from intelpick_interfaces.srv import MoveTo, SetGripper

from .roarm import RoArm


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm')
        self.arm = RoArm(
            port=self.declare_parameter('serial_port', '/dev/ttyUSB0').value,
            model=self.declare_parameter('model', 'm2').value,
            host=self.declare_parameter('host', '').value,
            dry_run=self.declare_parameter('dry_run', False).value,
            m3_pitch=self.declare_parameter('m3_pitch', 1.57).value,
        )
        self.speed = self.declare_parameter('speed', 0.25).value
        self.gripper_open = self.declare_parameter('gripper_open', 1.6).value
        self.gripper_closed = self.declare_parameter('gripper_closed', 3.1).value
        grip_torque = self.declare_parameter('grip_torque', 30).value  # % of servo max

        self.arm.connect()
        self.arm.set_grip_torque(grip_torque)
        self.pose_pub = self.create_publisher(PointStamped, '~/pose', 10)
        self.create_service(MoveTo, '~/move_to', self.on_move_to)
        self.create_service(SetGripper, '~/set_gripper', self.on_set_gripper)
        self.create_service(Trigger, '~/home', self.on_home)
        mode = 'DRY RUN' if self.arm.dry_run else (self.arm.host or self.arm.port)
        self.get_logger().info(f'RoArm-{self.arm.model.upper()} ready ({mode})')

    def on_move_to(self, req, res):
        spd = req.speed or self.speed
        res.success = self.arm.move_to(req.x * 1000, req.y * 1000, req.z * 1000, spd=spd)
        res.message = 'ok' if res.success else 'did not reach target (unreachable or timeout)'
        self.publish_pose()
        return res

    def on_set_gripper(self, req, res):
        angle = self.arm.set_gripper(self.gripper_open if req.open else self.gripper_closed)
        res.success, res.message = True, 'ok'
        res.angle = math.nan if angle is None else angle
        return res

    def on_home(self, req, res):
        res.success = self.arm.home()
        res.message = 'ok' if res.success else 'timeout'
        self.publish_pose()
        return res

    def publish_pose(self):
        p = self.arm.get_pose()
        if p is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'arm_base'
        msg.point.x, msg.point.y, msg.point.z = p.x / 1000, p.y / 1000, p.z / 1000
        self.pose_pub.publish(msg)

    def destroy_node(self):
        self.arm.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
