"""ROS2 utilities for calibration nodes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from charuco_board import decode_ros_image
from transforms import matrix_from_transform


class BaseCollectorNode(Node):
    """Base ROS2 node that subscribes to an image topic and provides TF lookups."""

    def __init__(self, image_topic: str, base_frame: str, gripper_frame: str, node_name: str = "calib_collector"):
        super().__init__(node_name)
        self.latest_image: Optional[np.ndarray] = None
        self.latest_stamp: Optional[float] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = base_frame
        self.gripper_frame = gripper_frame
        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.get_logger().info(f"Subscribed to {image_topic}; TF {base_frame} -> {gripper_frame}")

    def _image_cb(self, msg: Image) -> None:
        img = decode_ros_image(msg)
        if img is not None:
            self.latest_image = img
            self.latest_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def lookup_base_to_gripper(self) -> np.ndarray:
        t = self.tf_buffer.lookup_transform(self.base_frame, self.gripper_frame, rclpy.time.Time())
        return matrix_from_transform(t)
