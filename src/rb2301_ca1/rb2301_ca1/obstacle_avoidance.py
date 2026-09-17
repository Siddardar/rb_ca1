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
        self.last_heading = 0.0 # heading (deg) chosen last tick, so robot doesn't flip between equal gaps
        self.turn_dir = None # set a turn dir when stuck, kept until front clears

        self.SAFE_DIST = 0.4
        self.HEADINGS = np.arange(-90, 91, 5) # dir to check from center (min, max + 1, step size)
        self.SWITCH_PENALTY = 0.5 # cost per deg of changing heading; higher = more committed to last heading

        self.margin = 0.05
        self.ROBOT_WIDTH = 0.20 + self.margin
        self.ROBOT_LEN = 0.30 + self.margin

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
        scan = np.array(msg.ranges) # keep full scan
        # inf / nan / below-min readings all mean "nothing seen"; treat as max range so the maths stays finite
        bad = ~np.isfinite(scan) | (scan < msg.range_min)
        scan[bad] = msg.range_max
        self.last_scan = scan
        self.idx_per_deg = 1 / np.rad2deg(msg.angle_increment)

    def corridor(self, heading_deg, corridor_width):
        """Distance to nearest reading in a corridor along heading"""
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
        
    def footprint_width(self, heading_deg):
        """Width of the robot's footprint perpendicular to heading_deg (bounding box projection)"""
        rad = np.deg2rad(heading_deg)
        return self.ROBOT_WIDTH * abs(np.cos(rad)) + self.ROBOT_LEN * abs(np.sin(rad))

    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return # Does not run if the laser message is not received.

        # already turning: keep turning until straight ahead clears, ignore everything else
        if self.turn_dir is not None:
            if self.corridor(0, self.ROBOT_WIDTH) > self.SAFE_DIST:
                self.turn_dir = None
                self.last_heading = 0.0
            else:
                self.move_2D(0, 0, self.turn_dir * max_turn_velocity)
                return

        # clearance along every candidate heading
        clearance = np.array([self.corridor(h, self.footprint_width(h)) for h in self.HEADINGS])
        open_headings = self.HEADINGS[clearance > self.SAFE_DIST]

        # stuck: nothing open in the front half, commit to a turn dir
        if len(open_headings) == 0:
            left_room = np.max(clearance[self.HEADINGS > 0])
            right_room = np.max(clearance[self.HEADINGS < 0])
            self.turn_dir = 1 if left_room > right_room else -1
            self.get_logger().debug("stuck :( turning")
            self.move_2D(0, 0, self.turn_dir * max_turn_velocity)
            return

        # prefer straight ahead, but penalise flipping away from the heading chosen last -> prevents switching every tick
        cost = np.abs(open_headings) + self.SWITCH_PENALTY * np.abs(open_headings - self.last_heading)
        heading = float(open_headings[np.argmin(cost)])
        self.last_heading = heading

        self.get_logger().debug(f"heading: {heading:.0f} deg, clearance: {clearance[self.HEADINGS == heading][0]:.2f} m")

        # slide along the chosen heading without rotating -> ensures lidar alw points in same dir
        rad = np.deg2rad(heading)
        self.move_2D(max_translate_velocity * np.cos(rad), max_translate_velocity * np.sin(rad), 0)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()