import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from typing import NamedTuple  
import math

np.set_printoptions(
    2, suppress=True
)  # Print numpy arrays to specified d.p. and suppress scientific notation (e.g. 1e-5)

max_translate_velocity = 0.4 # Can be implemented as parameter
max_turn_velocity = max_translate_velocity * 2 # Can be implemented as parameter
set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

class Can(NamedTuple):
    bearing: float # degrees, +ve anticlockwise, 0 = straight ahead
    distance: float

class DoorWay(NamedTuple):
    left_can: Can
    right_can: Can
    gap: float # distance between the two readings

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
        self.cmd_turn = 0.0

        self.idx_per_deg = None # calc the number of indexes in scan array that correspond to 1 deg
        self.side = None # +1 = left, -1 = right; sticks until that side is fully blocked, then flips
        self.doorway = None # the gap we have committed to; while it is set nothing else runs
        self.drive_left = None # metres still to drive through the gap, set once we are lined up
        self.turned = 0.0 # net heading change since we committed, so we can undo it afterwards

        self.SAFE_DIST = 0.4

        self.margin = 0.05
        self.ROBOT_WIDTH = 0.20 + self.margin
        self.ROBOT_LEN = 0.22 + self.margin

        self.stop = False

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
        self.cmd_turn = float(turn)

        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        scan = np.array(msg.ranges) # keep full scan
        
        bad = ~np.isfinite(scan) | (scan < msg.range_min) # clean up values
        scan[bad] = msg.range_max

        self.last_scan = scan
        self.idx_per_deg = 1 / np.rad2deg(msg.angle_increment)

    def calculate_doorways(self, scan):
        """Pairs every can with the next one anticlockwise that sits at a similar distance, so each
        pair is two cans out of the same row with an opening between them"""

        SIMILAR = 0.2 # two cans this close in range are taken to be part of the same row
        MAX_SWEEP = 90 # a partner further round than this is on another wall, not beside this can

        cans = self.detect_cans(scan)
        doorways = []

        for idx, right in enumerate(cans):
            # cans come out anticlockwise, so walk that way to the next one at a similar distance.
            # anything skipped over sits in a different row and is seen through the opening, not in it
            left = None
            for ahead in range(1, len(cans)):
                candidate = cans[(idx + ahead) % len(cans)]
                if (candidate.bearing - right.bearing) % 360 > MAX_SWEEP: # gone past the end of the wall
                    break
                if abs(candidate.distance - right.distance) <= SIMILAR:
                    left = candidate
                    break

            if left is None: # nothing else at this distance, so this can is not one side of anything
                continue

            if abs(left.bearing) >= 90 and abs(right.bearing) >= 90: # at least one side has to be in front
                continue

            # cosine rule, with the angle swept anticlockwise from the right can round to the left one
            angle = math.radians((left.bearing - right.bearing) % 360)
            gap = math.sqrt(right.distance ** 2 + left.distance ** 2
                            - 2 * right.distance * left.distance * math.cos(angle))

            if gap < self.ROBOT_WIDTH: # too narrow to fit through
                continue

            doorways.append(DoorWay(left, right, gap))
        
        return doorways

    def detect_cans(self, scan):
        """Splits the scan where the range jumps, then keeps the groups that are nearer than their
        neighbours. Bearing is in degrees, +ve anticlockwise, 0 = straight ahead"""

        JUMP = 0.06 # a step bigger than this between neighbouring beams is the edge of an object

        def to_can(run):
            """A finished group becomes one can at its middle beam, bearing and range off the same reading"""
            middle = run[len(run) // 2]
            bearing = middle / self.idx_per_deg
            return Can(float(bearing - 360 if bearing > 180 else bearing), float(scan[middle]))

        cans = []
        step = np.abs(scan - np.roll(scan, 1)) # step[i] is the jump from beam i-1 to beam i

        # the scan wraps, so a can sat at 0 deg is split across both ends of the array. starting the
        # walk at the biggest jump means we begin on a real edge and every group comes out in one piece
        groups = []
        run = [] # indexes of the current group

        for angle in np.roll(np.arange(len(scan)), -int(np.argmax(step))):
            if run and step[angle] > JUMP: # a new object starts here
                groups.append(run)
                run = []
            run.append(angle)

        groups.append(run)

        if len(groups) < 2: # one unbroken surface all the way round, so nothing to pick out of it
            return cans

        # a can is a peak, so it is nearer than what sits either side of it. the empty space between
        # two cans is further than both and drops out here, without naming a max range anywhere
        nearest = [scan[g].min() for g in groups]

        for idx, group in enumerate(groups):
            if nearest[idx] > nearest[idx - 1] and nearest[idx] > nearest[(idx + 1) % len(groups)]:
                continue
            cans.append(to_can(group))

        return cans

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

    def doorway_frame(self, door):
        """The gap's midpoint and the line joining its two cans, both in robot coordinates"""
        points = np.array([[c.distance * math.cos(math.radians(c.bearing)),
                            c.distance * math.sin(math.radians(c.bearing))]
                           for c in (door.left_can, door.right_can)])
        return points.mean(axis=0), points[0] - points[1]

    def drive_through(self):
        """Drives straight through the gap, keeping to the middle of whatever is either side, then
        turns back onto the heading we came in on"""

        CENTRE_GAIN = 1.0 # the error here is a centimetre or two, nothing to be aggressive about
        LOOKAHEAD = 0.5 # only steer by cans in the channel we are in, not the next row
        TURN_RATE = 0.5 # rad/s, held flat so the turn back lands on the heading instead of creeping up on it
        TURN_TOL = np.deg2rad(1) # whatever is left here is heading error we never get back
        TURN_GAIN = 1.5

        if self.drive_left > 0: # still going through
            beside = [(c.distance * math.cos(math.radians(c.bearing)), c.distance * math.sin(math.radians(c.bearing)))
                      for c in self.detect_cans(self.last_scan)]
            beside = [y for x, y in beside if 0 < x < LOOKAHEAD]

            left = min((y for y in beside if y > 0), default=None)
            right = max((y for y in beside if y < 0), default=None)
            middle = (left + right) / 2 if left is not None and right is not None else 0.0 # halfway between them

            self.drive_left -= self.cmd_x * self.DT
            self.get_logger().debug(f"through: {self.drive_left:.3f} m left, off centre by {middle:+.3f}")
            self.move_2D(max_translate_velocity, CENTRE_GAIN * middle, 0)

        # proportional version, eases in but only ever gets within TURN_TOL. kept for the real robot
        # elif abs(self.turned) > TURN_TOL: # swing back onto the heading we came in on
        #     self.get_logger().debug(f"turning back: {math.degrees(self.turned):.1f} deg")
        #     self.move_2D(0, 0, -TURN_GAIN * self.turned)

        elif abs(self.turned) > 1e-6: # swing back onto the heading we came in on
            rate = min(TURN_RATE, abs(self.turned) / self.DT) # last tick turns exactly what is left
            self.get_logger().debug(f"turning back: {math.degrees(self.turned):.1f} deg")
            self.move_2D(0, 0, -math.copysign(rate, self.turned))

        else: # done, hand back to the normal behaviour
            self.get_logger().debug("through the gap")
            self.doorway, self.drive_left, self.turned = None, None, 0.0
            self.move_2D(0, 0, 0)

    def take_doorway(self):
        """Everything the robot does once a doorway is picked. Slides along the wall onto the normal
        through the gap's midpoint, then turns to face it. No corridor checks run while this is active"""

        MATCH = 0.10 # the midpoint only moves a strafe tick's worth between scans
        ALIGN_TOL = 0.05 # close enough to the normal to stop sliding
        TURN_TOL = np.deg2rad(3) # close enough to facing the gap to stop turning
        TURN_GAIN = 1.5
        SLIDE_GAIN = self.MAX_ACCEL / max_translate_velocity # hits full speed just as the ramp can still follow it down

        self.turned += self.cmd_turn * self.DT # net heading change, so the turn back can undo it

        if self.drive_left is not None: # lined up already, so we are on our way through
            return self.drive_through()

        # the stored reading is a tick out of date, so refresh it off this scan
        target, _ = self.doorway_frame(self.doorway)
        same = [(np.linalg.norm(self.doorway_frame(d)[0] - target), d)
                for d in self.calculate_doorways(self.last_scan)]
        same = [match for match in same if match[0] < MATCH]

        if same:
            self.doorway = min(same, key=lambda match: match[0])[1]

        # the normal runs through the midpoint at right angles to the line joining the two cans,
        # so how far off it we are is the midpoint measured along that line
        mid, chord = self.doorway_frame(self.doorway)
        chord = chord / np.linalg.norm(chord)
        offset = float(mid @ chord)

        turn = math.atan2(mid[1], mid[0]) # bearing to the midpoint, so zero means we are facing the gap

        self.get_logger().debug(f"repick: {self.doorway}")

        if abs(offset) > ALIGN_TOL: # slide along the wall onto the normal, easing off as it closes
            self.get_logger().debug(f"alligning: {offset}")
            self.move_2D(0, SLIDE_GAIN * offset * chord[1], 0)
        elif abs(turn) > TURN_TOL: # on the normal, so swing round to face the gap
            self.get_logger().debug(f"turning: {turn}")
            self.move_2D(0, 0, TURN_GAIN * turn)
        else: # lined up and pointing at it, so set off through
            self.drive_left = mid[0] + self.ROBOT_LEN + 0.30 # far enough for the tail to clear the cans
            self.get_logger().debug(f"lined up, driving {self.drive_left:.3f} m")
            self.move_2D(0, 0, 0)

    def timer_callback(self):
        """Controller loop"""

        if self.stop:
            return self.move_2D(0,0,0)


        if self.last_scan is None:
            return # Does not run if the laser message is not received.

        if self.doorway is not None: # committed to a gap, so nothing else gets a say
            self.take_doorway()
            return

        # strip ahead is as wide as the robot, strips to the sides are as long as the robot
        front = self.corridor(0, self.ROBOT_WIDTH)
        left = self.corridor(90, self.ROBOT_LEN)
        right = self.corridor(-90, self.ROBOT_LEN)

        # front clear: just go forward
        if front > self.SAFE_DIST:
            #self.get_logger().debug(f"forward, front: {front:.2f} m")
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
            #self.get_logger().debug(f"side blocked, switching to {'L' if self.side > 0 else 'R'}")

        # both sides blocked: go to side has more room and re-check next tick
        if room <= self.SAFE_DIST:
            self.side = 1 if left > right else -1
            room = left if self.side > 0 else right
            #self.get_logger().debug(f"both blocked, recheck {'L' if self.side > 0 else 'R'} (left: {left:.2f}, right: {right:.2f})")

        DOORWAY_DIST = 0.5 # far enough to see a gap in the blocking row coming, near enough to rule out the next row

        # commit to a doorway on the side we are strafing towards once it is near enough to take
        for door in self.calculate_doorways(self.last_scan):
            mid, _ = self.doorway_frame(door)

            if mid[1] * self.side > 0 and np.linalg.norm(mid) < DOORWAY_DIST:
                self.get_logger().debug(f"picked: {door}")
                self.doorway = door
                self.take_doorway()
                return

        #self.get_logger().debug(f"strafe {'L' if self.side > 0 else 'R'}, room: {room:.2f} m, front: {front:.2f} m")
        self.move_2D(0, self.side * max_translate_velocity, 0)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
