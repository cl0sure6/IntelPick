"""USB webcam -> /camera/image_raw. Plain OpenCV so it works on a bare ROS 2 install."""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera')
        device = self.declare_parameter('device', '/dev/video0').value
        width = self.declare_parameter('width', 640).value
        height = self.declare_parameter('height', 480).value
        fps = self.declare_parameter('fps', 15.0).value
        self.frame_id = self.declare_parameter('frame_id', 'camera').value

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        # MJPG keeps USB bandwidth low, which matters when the camera is forwarded over usbipd.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always hand out the newest frame
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open camera {device}')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, 'camera/image_raw', 1)
        self.timer = self.create_timer(1.0 / fps, self.grab)
        self.get_logger().info(f'camera {device} {width}x{height} @ {fps} fps')

    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('frame grab failed', throttle_duration_sec=2.0)
            return
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
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
