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

        self.DT = 0.05 # controller period (s)
        self.timer = self.create_timer(self.DT, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.MAX_ACCEL = 0.8 # m/s^2 per axis; lower = gentler direction changes (0.4 m/s -> 0 takes 0.4 / MAX_ACCEL s)
        self.cmd_x = 0.0 # velocity actually sent last tick, so we can ramp towards the new target
        self.cmd_y = 0.0

        self.idx_per_deg = None # calc the number of indexes in scan array that correspond to 1 deg
        self.side = None # +1 = left, -1 = right; sticks until that side is fully blocked, then flips

        self.SAFE_DIST = 0.4

        self.margin = 0.05
        self.ROBOT_WIDTH = 0.20 + self.margin
        self.ROBOT_LEN = 0.30 + self.margin

    def regularise(self, target, current):
        """Move current towards target by at most MAX_ACCEL * DT, so velocity ramps instead of jumping"""
        step = self.MAX_ACCEL * self.DT
        return float(np.clip(target, current - step, current + step))

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0):
        """Publishes a twist command to move in 2D space. +ve x is forwards, +ve y is left, and +ve turn is anticlockwise.
        x and y are rate limited: a direction change ramps one axis down while the other ramps up, so the robot slows through the switch rather than stopping or jerking"""
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_turn_velocity, max_turn_velocity)

        x = self.cmd_x = self.regularise(x, self.cmd_x)
        y = self.cmd_y = self.regularise(y, self.cmd_y)

        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)

        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        scan = np.array(msg.ranges) # keep full scan
        
        bad = ~np.isfinite(scan) | (scan < msg.range_min) # clean up values
        scan[bad] = msg.range_max

        self.last_scan = scan
        self.idx_per_deg = 1 / np.rad2deg(msg.angle_increment)

    def corridor(self, heading_deg, corridor_width):
        """Distance to nearest reading in a corridor along heading (0 = front, +ve = anticlockwise)"""
        n = len(self.last_scan)

        # angles (in rad for np.sin/np.cos) in corridor dir
        angles = np.deg2rad(np.arange(n) / self.idx_per_deg - heading_deg) 

        fwd = self.last_scan * np.cos(angles)
        lat = self.last_scan * np.sin(angles)

        # cos 90 - 270 < 0 => fwd > 0 is what we care about since angles are relative to corridor dir anyway
        # also if lateral dis < width / 2 its in path
        in_path = (fwd > 0) & (np.abs(lat) < corridor_width / 2) # boolean arr

        # fwd[in_path] acts like a boolean mask keeping only values in fwd that have a corresponding True in in_path
        # returns the closest object in dir of corridor if any exists
        return np.min(fwd[in_path]) if in_path.any() else np.inf

    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return # Does not run if the laser message is not received.

        # strip ahead is as wide as the robot, strips to the sides are as long as the robot
        front = self.corridor(0, self.ROBOT_WIDTH)
        left = self.corridor(90, self.ROBOT_LEN)
        right = self.corridor(-90, self.ROBOT_LEN)

        # front clear: just go forward
        if front > self.SAFE_DIST:
            self.get_logger().debug(f"forward, front: {front:.2f} m")
            self.move_2D(max_translate_velocity, 0, 0)
            return

        # first time blocked: pick the side with more room, after that stick with it
        if self.side is None:
            self.side = 1 if left > right else -1

        # committed side fully blocked: flip to the other side
        room = left if self.side > 0 else right
        if room <= self.SAFE_DIST:
            self.side = -self.side
            room = left if self.side > 0 else right
            self.get_logger().debug(f"side blocked, switching to {'L' if self.side > 0 else 'R'}")

        # both sides blocked: go to side has more room and re-check next tick
        if room <= self.SAFE_DIST:
            self.side = 1 if left > right else -1
            room = left if self.side > 0 else right
            self.get_logger().debug(f"both blocked, recheck {'L' if self.side > 0 else 'R'} (left: {left:.2f}, right: {right:.2f})")

        self.get_logger().debug(f"strafe {'L' if self.side > 0 else 'R'}, room: {room:.2f} m, front: {front:.2f} m")
        self.move_2D(0, self.side * max_translate_velocity, 0)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
