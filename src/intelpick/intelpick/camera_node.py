"""Webcam or Wi-Fi camera -> /camera/image_raw. Plain OpenCV so it works on a bare ROS 2 install."""

import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image

from .camera_source import FrameSource


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera')
        # dynamic_typing so `device: 0` in YAML works as well as a path or URL.
        device = self.declare_parameter(
            'device', '/dev/video0', ParameterDescriptor(dynamic_typing=True)).value
        width = self.declare_parameter('width', 640).value
        height = self.declare_parameter('height', 480).value
        fps = self.declare_parameter('fps', 15.0).value
        self.frame_id = self.declare_parameter('frame_id', 'camera').value

        self.source = FrameSource(device, width, height,
                                  log=lambda s: self.get_logger().warn(s))
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, 'camera/image_raw', 1)
        self.last_frame = self.get_clock().now()
        self.timer = self.create_timer(1.0 / fps, self.grab)
        kind = 'stream' if self.source.is_stream else f'{width}x{height}'
        self.get_logger().info(f'camera {self.source.name} ({kind}) @ {fps} fps')

    def grab(self):
        frame = self.source.read()
        now = self.get_clock().now()
        if frame is None:
            if (now - self.last_frame).nanoseconds > 3e9:
                self.get_logger().warn('no frames for 3 s', throttle_duration_sec=5.0)
            return
        self.last_frame = now
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self.source.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
