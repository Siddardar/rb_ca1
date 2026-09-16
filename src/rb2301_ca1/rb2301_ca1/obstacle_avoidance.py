import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(
    2, suppress=True
)  # Print numpy arrays to specified d.p. and suppress scientific notation (e.g. 1e-5)

max_translate_velocity = 0.4 # Can be implemented as parameter
max_turn_velocity = max_translate_velocity * 2 # Can be implemented as parameter
set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting Obstacle Avoidance")

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)  # Publish to cmd_vel node
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # The subscriber to the Lidar ranges.
        self.last_scan = None # Copied laser scan message

        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.idx_per_deg = None # calc the number of indexes in scan array that correspond to 1 deg
        self.horizontal_dir = None # set a horizontal dir so robot doesn't oscilate
        self.turn_dir = None # set a turn dir when stuck, kept until front clears

        self.ANGLE = 30 # angle of cone
        self.SAFE_DIST = 0.4

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0):
        """Publishes a twist command to move in 2D space. +ve x is forwards, +ve y is left, and +ve turn is anticlockwise"""
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.last_scan = np.array(msg.ranges) # keep full scan
        self.idx_per_deg = 1 / np.rad2deg(msg.angle_increment)

    def cone(self, center_deg, angle_deg):
        """Return scan readings within +- angle of center"""
        low = int((center_deg - angle_deg) * self.idx_per_deg) 
        high = int((center_deg + angle_deg) * self.idx_per_deg)

        idx = np.arange(low, high + 1) % int(360 * self.idx_per_deg)

        return self.last_scan[idx]

    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return # Does not run if the laser message is not received.
        
        ######################## MODIFY CODE HERE ########################

        front_cone = self.cone(0, self.ANGLE)
        left_cone = self.cone(90, self.ANGLE)
        right_cone = self.cone(270, self.ANGLE)

        min_front = np.min(front_cone)
        min_left = np.min(left_cone)
        min_right = np.min(right_cone)  

        self.get_logger().debug(f"min_front: {min_front}, min_left: {min_left}, min_right: {min_right}")

        # move forward as much as possible
        if min_front > self.SAFE_DIST:
            self.horizontal_dir = None
            self.turn_dir = None
            self.move_2D(max_translate_velocity, 0, 0)
            return 

        # already turning: keep turning until front clears, ignore sides
        if self.turn_dir is not None:
            self.move_2D(0, 0, self.turn_dir * max_turn_velocity)
            return

        # if curr horizontal dir is blocked, reset
        if self.horizontal_dir is not None:
            chosen_dir = min_left if self.horizontal_dir == 1 else min_right
            if chosen_dir <= self.SAFE_DIST:
                self.horizontal_dir = None

        # pick best possible horizontal dir
        if self.horizontal_dir is None:
            if max(min_left, min_right) > self.SAFE_DIST:
                self.horizontal_dir = 1 if min_left > min_right else -1

        if self.horizontal_dir is not None:
            self.move_2D(0, self.horizontal_dir * max_translate_velocity, 0)
            return

        # stuck: turn and see if can find better path and go straight
        self.get_logger().debug("stuck :(")
        self.horizontal_dir = None
        self.turn_dir = 1 if min_left > min_right else -1
        self.move_2D(0, 0, self.turn_dir * max_turn_velocity)

        ######################## MODIFY CODE HERE ########################


def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()