import itertools
import math
from collections import Counter
from typing import NamedTuple

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

CAN_RADIUS = 0.05 # cans are known cylinders, so a cluster of beams can be turned back into a centre

# Where the lidar sits relative to the point the robot turns about. Everything the scan
# gives us is measured from the lidar, but the robot swings about its own centre, so the
# lidar rides round on the end of this arm. Get this wrong and a rotation measured from
# how far a nearby can appears to move comes out short, and the robot over-turns. Set
# both to zero if the lidar really is at the turn centre.
LIDAR_FORWARD = 0.11423
LIDAR_LEFT = 0.0


def to_body(point):
    """A point measured by the lidar, moved into the frame the robot turns in."""
    return np.asarray(point, dtype=float) + np.array([LIDAR_FORWARD, LIDAR_LEFT])


def wrap_angle(angle):
    """Fold an angle into -pi..pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Scan(NamedTuple):
    """One laser scan reduced to what the geometry below needs"""
    ranges: np.ndarray # raw metres, invalid entries left untouched
    valid: np.ndarray # True where the beam is a real return
    bearings: np.ndarray # radians, +ve anticlockwise from straight ahead; beam 0 is dead ahead
    step: float # angular spacing between beams, always positive
    closed: bool # the scan covers a full turn, so the two ends of the array are adjacent


class Can(NamedTuple):
    """One detected can. Distance is to the can CENTRE, not to its surface."""
    bearing: float
    distance: float
    beams: int
    chord: float # length of the visible arc, used to tell a can from a wall
    half_left: float # angular extent of the cluster to the left of bearing
    half_right: float # and to the right of it, so the mouth check can skip the can itself


class Doorway(NamedTuple):
    """A pair of cans with a usable opening between them.

    Only openings the robot could actually drive through are built, so there is no
    validity flag to test: if a Doorway exists then it is wide enough and its mouth
    is empty.
    """
    left: Can
    right: Can
    separation: float # centre to centre
    gap: float # clear space between the two surfaces, i.e. what the robot must fit through
    bearing: float # to the midpoint of the doorway
    midpoint: np.ndarray # middle of the opening, +x forward and +y left


def prepare_scan(ranges, angle_increment, range_min, range_max):
    """Turn one LaserScan into a Scan.

    Beam 0 points straight ahead and each following beam is one angle_increment
    further anticlockwise, which is the same convention corridor() uses. The
    message's angle_min is deliberately ignored: this robot reports a value that
    does not match how the lidar is actually fitted, and trusting it turns every
    bearing 180 degrees the wrong way.

    Invalid beams are recorded in a mask rather than overwritten with the maximum
    range. A filled-in value looks like a real surface, which would both merge
    clusters that should stay separate and hide the empty space that is the only
    evidence a doorway is actually open.
    """
    r = np.asarray(ranges, dtype=float)
    step = abs(float(angle_increment))
    valid = np.isfinite(r) & (r >= range_min) & (r < range_max)
    bearings = wrap_angle(np.arange(r.size) * step)
    closed = bool(r.size and step > 0.0 and abs(r.size * step - 2 * np.pi) < step)
    return Scan(r, valid, bearings, step, closed)


def centre_of(can):
    """Cartesian position of a can centre, +x forward and +y left"""
    return np.array([can.distance * np.cos(can.bearing), can.distance * np.sin(can.bearing)])


def detect_cans(scan, radius=CAN_RADIUS, cluster_step=0.06, min_beams=2,
                max_distance=2.0, max_bearing=np.deg2rad(120.0)):
    """Cluster one scan into cans and return them as centres.

    A beam only ever reports the near face of a can, so every cluster is turned
    into a centre by adding the radius along its bearing. Measuring between near
    faces instead would overstate the space between two cans by roughly
    2 * radius * (1 - radius / distance), about 8cm at 1m, far more than the
    margin the robot has.

    Clusters are cut wherever a beam is invalid or the range steps by more than
    cluster_step. Anything whose visible arc is longer than a can could be, or
    that is far enough away for the beam spacing to make its centre unreliable,
    is dropped rather than guessed at.
    """
    if scan.ranges.size == 0 or scan.step == 0.0:
        return []
    index = np.flatnonzero(scan.valid)
    if index.size == 0:
        return []

    # Split the valid beams into runs that are both adjacent and at a similar range.
    cuts = [0]
    for k in range(1, index.size):
        previous, current = index[k - 1], index[k]
        if current != previous + 1 or abs(scan.ranges[current] - scan.ranges[previous]) > cluster_step:
            cuts.append(k)
    clusters = [index[a:b] for a, b in zip(cuts, cuts[1:] + [index.size])]

    # A full turn closes on itself, so a can sitting on the seam arrives as two runs.
    n = scan.ranges.size
    if scan.closed and len(clusters) > 1:
        first, last = clusters[0], clusters[-1]
        if first[0] == 0 and last[-1] == n - 1 and abs(scan.ranges[0] - scan.ranges[n - 1]) <= cluster_step:
            clusters[0] = np.concatenate((last, first))
            clusters.pop()

    cans = []
    for cluster in clusters:
        if cluster.size < min_beams:
            continue
        # Unwrap before averaging so a cluster spanning the seam still averages sanely.
        spread = np.unwrap(scan.bearings[cluster])
        bearing = wrap_angle(float(spread.mean()))
        surface = float(scan.ranges[cluster].min()) # closest point of a circle lies on the line to its centre
        distance = surface + radius
        if distance > max_distance:
            continue
        # Kept deliberately wider than the forward half. A can a little behind the
        # robot can still be one side of an opening that is in front of it, so
        # whether an opening is ahead is decided per opening, not per can.
        if abs(bearing) > max_bearing:
            continue
        low, high = float(spread.min()), float(spread.max())
        ends = np.array([[scan.ranges[i] * np.cos(b), scan.ranges[i] * np.sin(b)]
                         for i, b in ((cluster[0], spread[0]), (cluster[-1], spread[-1]))])
        chord = float(np.linalg.norm(ends[1] - ends[0]))
        # A can's silhouette is at most a diameter wide; allow the beam spacing on top.
        if chord > 2 * radius + 2 * distance * scan.step:
            continue
        cans.append(Can(bearing, distance, int(cluster.size), chord,
                        high - float(spread.mean()), float(spread.mean()) - low))
    return cans


def mouth_blocking_range(scan, left, right, clear_margin=0.05):
    """Range of the nearest return sitting in the mouth between two cans, else inf.

    Two cans being neighbours in bearing only proves no other CAN was detected
    between them. Anything the can detector rejected, a wall stub or a box or a
    chair leg, is still in the scan but absent from the can list, so without this
    check a plugged doorway reports exactly the same width as an open one.

    Only beams strictly between the two clusters are considered, and a return
    counts as blocking when it lies on the robot's side of the line joining the
    two centres. Obstructions deeper than that line are the crossing's problem,
    not the doorway's.
    """
    start = right.bearing + right.half_left
    window = (left.bearing - left.half_right) - start
    if not 0.0 < window < np.pi:
        return 0.0 # the cans overlap in bearing or straddle the rear: not an opening at all
    guard = scan.step # keep the cans' own beams out of the window
    offset = wrap_angle(scan.bearings - start)
    inside = scan.valid & (offset > guard) & (offset < window - guard)
    if not np.any(inside):
        return np.inf
    r, b = scan.ranges[inside], scan.bearings[inside]
    points = np.column_stack((r * np.cos(b), r * np.sin(b)))
    first, second = centre_of(left), centre_of(right)
    chord = second - first
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return 0.0
    normal = np.array([-chord[1], chord[0]]) / length
    if normal @ ((first + second) / 2) < 0.0:
        normal = -normal # point away from the robot
    depth = (points - first) @ normal
    blocking = depth < clear_margin
    return float(r[blocking].min()) if np.any(blocking) else np.inf


def doorway_normal(doorway):
    """Unit vector the robot must face to drive through the doorway, in its own frame."""
    chord = centre_of(doorway.right) - centre_of(doorway.left)
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    across = chord / length
    normal = np.array([-across[1], across[0]])
    if normal @ doorway.midpoint < 0.0:
        normal = -normal # point away from the robot, the way it would drive through
    return normal, across


def doorway_alignment(doorway):
    """Where the robot stands relative to a doorway, as (offset, approach, across).

    A doorway is driven through along the normal to the line joining its two cans,
    so being near the opening is not enough: the robot has to sit on that normal.
    `offset` is how far it is off that centre line, measured along the chord, and is
    what has to be driven to zero. `approach` is what is left to run along the
    normal once it is lined up. `across` is the chord direction, which says which
    way to strafe to reduce the offset.
    """
    normal, across = doorway_normal(doorway)
    middle = to_body(doorway.midpoint) # the body has to sit on the centre line, not the lidar
    return float(middle @ across), float(middle @ normal), across


def pair_doorways(cans, scan, min_gap, radius=CAN_RADIUS, clear_margin=0.05,
                  max_bearing=np.deg2rad(90.0)):
    """Find every pair of cans with a usable opening between them.

    Every pair is considered, not just cans that are neighbours in bearing. A
    tilted opening has one can nearer than the other, so you look past the near
    one and see the wall carrying on behind it, which puts unrelated cans between
    the real pair in bearing order. Pairing neighbours only would never build the
    real opening as a candidate at all.

    Four things have to hold, and each throws out a different kind of nonsense:

      width       the robot has to fit, so the gap must beat min_gap.
      ahead       the opening itself must be in front of the robot. This is tested
                  on the opening, not on its two cans: a can a couple of degrees
                  behind the robot can still form an opening well in front of it,
                  and testing the cans instead silently loses those openings.
      nothing nearer  the two cans must be what actually forms the opening. If
                  some other can sits closer to the middle than they do, they are
                  not its edges.

    There is deliberately no test that the two cans are at a similar range. It is
    tempting, because pairing a near can with a far one down the line of sight is
    a common way to invent an opening that is not there. But a tilted opening has
    one can nearer than the other by construction, and the more squarely you face
    it the bigger that difference gets, so such a test throws away exactly the
    openings worth finding.
      clear mouth the space between them must be empty, which only the beams can
                  answer. See mouth_blocking_range.

    Returns the openings, plus a tally of why the rest were dropped so a log can
    say what happened instead of going quiet.
    """
    doorways, rejected = [], Counter()
    for i, j in itertools.combinations(range(len(cans)), 2):
        first, second = cans[i], cans[j]
        left, right = (first, second) if first.bearing > second.bearing else (second, first)
        delta = wrap_angle(left.bearing - right.bearing)
        separation = math.sqrt(max(0.0, left.distance ** 2 + right.distance ** 2
                                   - 2 * left.distance * right.distance * math.cos(delta)))
        gap = separation - 2 * radius
        if gap < min_gap:
            rejected["too narrow"] += 1
            continue
        midpoint = (centre_of(left) + centre_of(right)) / 2
        if abs(math.atan2(midpoint[1], midpoint[0])) > max_bearing:
            rejected["not in front of the robot"] += 1
            continue
        if any(k != i and k != j
               and float(np.linalg.norm(centre_of(cans[k]) - midpoint)) < separation / 2 - 1e-9
               for k in range(len(cans))):
            rejected["another can sits nearer its middle"] += 1
            continue
        blocking = mouth_blocking_range(scan, left, right, clear_margin)
        if np.isfinite(blocking):
            rejected["mouth blocked"] += 1
            continue
        doorways.append(Doorway(left, right, separation, gap,
                                float(math.atan2(midpoint[1], midpoint[0])), midpoint))
    return doorways, rejected


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

        self.scan = None # the same scan kept as geometry: ranges, validity mask and bearings
        self.scan_seq = 0 # counted so a doorway is confirmed over scans, not over controller ticks
        self.checked_seq = -1 # the scan the doorway search last ran on
        self.MIN_GAP = self.ROBOT_WIDTH # an opening narrower than the robot is not a way through
        self.DOORWAY_CONFIRM = 3 # scans the same doorway must appear on before we act on it
        self.DOORWAY_MATCH = 0.12 # m the midpoint may move between scans and still count as the same doorway
        self.ALIGN_TOL = 0.05 # m off the doorway's centre line that still counts as lined up
        self.APPROACH_MAX = 0.60 # m; any further from the doorway than this and it keeps closing
        self.APPROACH_MIN = 0.15 # m; level with an opening or past it, there is no driving through
                                 # it, so it is not worth sliding sideways towards
        self.GIVE_UP_TICKS = int(3.0 / self.DT) # after an opening turns out to be unreachable, spend
                                                # this long searching before looking at openings again
        self.give_up = 0 # ticks left of that
        self.MIN_SIDE_ROOM = 0.24 # m; a corner of the robot reaches 0.18 m from its centre and a
                                  # can adds 0.05 m, so strafing towards anything nearer than this
                                  # drives a corner into it
        self.CLOSE_CLEARANCE = 0.26 # m of side room to keep while closing on a doorway. SAFE_DIST
                                    # is the limit for wandering blind; lining up on an opening next
                                    # to a wall needs to get nearer than that. It cannot go much
                                    # below this: the robot's corners reach 0.18 m from its centre,
                                    # so anything under about 0.23 m lets a corner catch a can.
        self.doorways = [] # [doorway, scans it has been seen on] from the previous scan
        self.facing = None # the one doorway we committed to, once lined up with it
        self.turn_target = 0.0 # rad to turn, as measured when we committed
        self.turn_done = 0.0 # rad of that turn already commanded
        self.facing_midpoint = None # where the committed opening should be now, in the robot's frame
        self.crossing = False # driving through the opening
        self.hug_at = None # where the can we are keeping station on should be now
        self.hug_side = 1.0 # +1 if that can is on our left, -1 if on our right
        self.cross_ticks = 0
        self.HUG = 0.18 # m from that can's centre: 0.10 of robot plus 0.05 of can plus 0.03 spare
        self.HUG_MATCH = 0.15 # m a can may be from where we expect it and still be the same can
        self.CROSS_SPEED = 0.15 # m/s through the gap
        self.CROSS_STOP = 0.15 # m of front clearance below which we stop no matter what
        self.CROSS_TIMEOUT = 10.0 # s before giving up and stopping
        self.CLEAR_BEHIND = 0.25 # m the hugged can must be behind us before we dare turn: the robot
                                 # sweeps a 0.18 m circle turning on the spot and half the gap is
                                 # only 0.16 m, so it cannot rotate until it is properly clear
        self.turning_back = False # rotating back onto the heading we started with
        self.back_bearing = 0.0 # bearing of the hugged can when the turn back began
        self.back_ticks = 0
        self.turn_ticks = 0 # how long the turn has been going, in controller ticks
        self.TURN_TOL = np.deg2rad(3.0) # heading error that counts as facing the doorway
        self.TURN_GAIN = 2.0 # rad/s of turn per rad of heading error
        self.TURN_TIMEOUT = 6.0 # s before giving up and stopping where it is
        self.halted = False # a doorway has been reported, so hold still

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

        # corridor() wants a missing beam to read as far away, while clustering cans needs
        # to know it is missing at all, so keep the untouched scan beside the filled one.
        self.scan = prepare_scan(msg.ranges, msg.angle_increment,
                                 msg.range_min, msg.range_max)
        self.scan_seq += 1

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

    def doorway_candidates(self):
        """Openings on the latest scan, each with the run of scans it has been seen on.

        Counting scans rather than controller ticks matters: the timer runs faster
        than the lidar, so counting ticks would let a single bad scan confirm itself.
        Between scans the previous list is handed back unchanged, so steering stays
        smooth instead of stuttering at the lidar rate.
        """
        if self.scan is None or self.scan_seq == self.checked_seq:
            return self.doorways
        self.checked_seq = self.scan_seq

        cans = detect_cans(self.scan)
        doorways, rejected = pair_doorways(cans, self.scan, self.MIN_GAP)
        if not doorways:
            self.doorways = []
            reasons = ", ".join(f"{count} {why}" for why, count in rejected.most_common())
            self.get_logger().debug(
                f"no doorway: {len(cans)} cans" + (f"; rejected {reasons}" if reasons else ""))
            return self.doorways

        # Carry each opening's run of sightings forward by matching on midpoint.
        tracked = []
        for door in doorways:
            streak = 1
            for old, old_streak in self.doorways:
                if float(np.linalg.norm(door.midpoint - old.midpoint)) <= self.DOORWAY_MATCH:
                    streak = old_streak + 1
                    break
            tracked.append([door, streak])
        self.doorways = tracked

        self.get_logger().debug("doorways: " + "; ".join(
            f"gap {door.gap:.3f} m at {np.rad2deg(door.bearing):+.1f} deg, seen on "
            f"{streak} of {self.DOORWAY_CONFIRM} scans" for door, streak in tracked))
        return tracked

    def face_doorway(self):
        """Turn in place until the robot points along the committed doorway's normal.

        Only the opening the robot committed to is ever considered. Nothing else can
        take over part way through a turn the robot has already positioned itself
        for, however inviting it looks.

        That one opening is measured again in every new scan, and the heading error
        comes from the normal as it reads now. Counting up the turn commands instead
        does not work: the base does not rotate as far as it is asked to, so the node
        believes the turn is finished while the robot is still tens of degrees short.
        The lidar is the only thing here that can see how far it has actually turned.

        To find the same opening again while the view is rotating, its expected
        position is carried round with the robot. If it is lost for a moment, the
        turn falls back to whatever is left of the angle measured at commit time.
        """
        self.turn_ticks += 1
        if abs(self.cmd_x) > 0.01 or abs(self.cmd_y) > 0.01:
            self.move_2D(0, 0, 0) # still sliding: settle before turning on the spot
            return

        measured = None
        for door, _ in self.doorway_candidates():
            if float(np.linalg.norm(door.midpoint - self.facing_midpoint)) <= self.DOORWAY_MATCH:
                measured = door
                break

        if measured is not None:
            self.facing = measured
            self.facing_midpoint = measured.midpoint.copy()
            normal, _ = doorway_normal(measured)
            error = math.atan2(float(normal[1]), float(normal[0]))
        else:
            error = self.turn_target - self.turn_done

        if self.turn_ticks * self.DT > self.TURN_TIMEOUT and abs(error) > 3 * self.TURN_TOL:
            # Never finished lining up. Driving through on this heading would be a guess,
            # so let the opening go and look again.
            self.get_logger().info(
                f"turn timed out {np.rad2deg(error):+.1f} deg short; giving the opening up")
            self.facing, self.facing_midpoint = None, None
            self.doorways, self.give_up = [], self.GIVE_UP_TICKS
            self.move_2D(0, 0, 0)
            return

        if abs(error) <= self.TURN_TOL or self.turn_ticks * self.DT > self.TURN_TIMEOUT:
            if abs(error) > self.TURN_TOL:
                self.get_logger().info(
                    f"turn finished {np.rad2deg(error):+.1f} deg out; close enough to drive")
            offset, approach, _ = doorway_alignment(self.facing)
            confirmed = [door for door, streak in self.doorways
                         if streak >= self.DOORWAY_CONFIRM] or [self.facing]
            self.report_doorways(confirmed, self.facing)
            self.get_logger().info(
                f"facing the doorway: heading error {np.rad2deg(error):+.2f} deg, off its "
                f"centre line by {offset:+.3f} m, {approach:.3f} m to run through it "
                f"(commanded {np.rad2deg(self.turn_done):+.1f} deg to get here)")
            # Hug the can on the outside of the turn on the way through: turn left and
            # that is the right hand one.
            turning_left = self.turn_target > 0.0
            hug_can = self.facing.right if turning_left else self.facing.left
            self.hug_side = -1.0 if turning_left else 1.0
            self.hug_at = centre_of(hug_can)
            self.crossing, self.cross_ticks = True, 0
            self.get_logger().info(
                f"driving through, keeping {self.HUG:.2f} m off the "
                f"{'right' if turning_left else 'left'} hand can")
            self.move_2D(0, 0, 0)
            return

        turn = float(np.clip(self.TURN_GAIN * error, -max_turn_velocity, max_turn_velocity))
        self.turn_done += turn * self.DT
        # Carry the opening round with the robot, so it can be found again next scan.
        step = -turn * self.DT
        cs, sn = math.cos(step), math.sin(step)
        x, y = float(self.facing_midpoint[0]), float(self.facing_midpoint[1])
        self.facing_midpoint = np.array([x * cs - y * sn, x * sn + y * cs])
        self.get_logger().debug(
            f"turning to face the doorway: {np.rad2deg(error):+.1f} deg to go"
            + ("" if measured is not None else ", opening not seen this scan"))
        self.move_2D(0, 0, turn)

    def cross_doorway(self):
        """Drive through the opening, keeping station on one can, and stop once past it.

        Only one can is tracked, the one on the outside of the turn. Once the robot is
        between the two, the other falls behind the beam fan or is shadowed by the
        robot itself, so a single can held at a known distance is the only thing left
        that still works.

        That can is carried along with the robot's own motion between scans and
        snapped back onto the nearest real can whenever a new scan arrives. The robot
        is through when it has gone past the robot's own tail.
        """
        self.cross_ticks += 1
        self.hug_at = self.hug_at - np.array([self.cmd_x, self.cmd_y]) * self.DT

        if self.scan is not None and self.scan_seq != self.checked_seq:
            self.checked_seq = self.scan_seq
            # All round, not the usual forward cone: the whole point is to keep watching
            # this can as it goes past and ends up behind. Lose it and the only thing left
            # is counting up commanded motion, which this base does not deliver.
            cans = detect_cans(self.scan, max_bearing=np.pi)
            if cans:
                seen = min(cans, key=lambda can: float(np.linalg.norm(centre_of(can) - self.hug_at)))
                if float(np.linalg.norm(centre_of(seen) - self.hug_at)) <= self.HUG_MATCH:
                    self.hug_at = centre_of(seen)

        ahead = self.corridor(0, self.ROBOT_WIDTH)
        behind = to_body(self.hug_at)
        through = behind[0] < -self.CLEAR_BEHIND
        if through or ahead < self.CROSS_STOP or self.cross_ticks * self.DT > self.CROSS_TIMEOUT:
            why = ("through the gap" if through else
                   f"something {ahead:.2f} m ahead" if ahead < self.CROSS_STOP else
                   "crossing timed out")
            self.get_logger().info(
                f"{why}. The can we kept station on is now {self.hug_at[0]:+.3f} m ahead "
                f"and {self.hug_at[1]:+.3f} m left")
            self.crossing = False
            self.move_2D(0, 0, 0)
            if not through:
                # Something got in the way that was not there when we committed, or it ran
                # out of time. Give the opening up and go back to searching rather than
                # sitting here for good: stopping is not an outcome, it is a dead end.
                self.facing, self.facing_midpoint, self.hug_at = None, None, None
                self.doorways, self.give_up = [], self.GIVE_UP_TICKS
                return
            self.turning_back, self.back_ticks = True, 0
            start = to_body(self.hug_at)
            self.back_bearing = math.atan2(float(start[1]), float(start[0]))
            self.get_logger().info(
                f"through; turning {np.rad2deg(-self.turn_target):+.1f} deg back onto the "
                f"heading we started on")
            return

        error = float(behind[1]) - self.hug_side * self.HUG
        sideways = float(np.clip(2.0 * error, -self.CROSS_SPEED, self.CROSS_SPEED))
        self.get_logger().debug(
            f"crossing: can {self.hug_at[0]:+.3f} m ahead, {self.hug_at[1]:+.3f} m left, "
            f"{error:+.3f} m off the distance to hold, front {ahead:.2f} m")
        self.move_2D(self.CROSS_SPEED, sideways, 0)

    def turn_back(self):
        """Rotate back onto the heading the robot set out on, then carry on forwards.

        Measured off the can it just hugged rather than counted off the commands. The
        robot turns on the spot here, so that can's bearing shifts by exactly minus
        however far the robot has turned, which makes one can enough to close the loop
        on rotation. Its expected position is carried round as the view rotates so it
        can be picked out again in each scan.
        """
        self.back_ticks += 1
        if abs(self.cmd_x) > 0.01 or abs(self.cmd_y) > 0.01:
            self.move_2D(0, 0, 0) # settle before turning on the spot
            return

        if self.scan is not None and self.scan_seq != self.checked_seq:
            self.checked_seq = self.scan_seq
            cans = detect_cans(self.scan, max_bearing=np.pi)
            if cans:
                seen = min(cans, key=lambda can: float(np.linalg.norm(centre_of(can) - self.hug_at)))
                if float(np.linalg.norm(centre_of(seen) - self.hug_at)) <= self.HUG_MATCH:
                    self.hug_at = centre_of(seen)

        seen = to_body(self.hug_at) # bearing about the turn centre, not about the lidar
        turned = wrap_angle(self.back_bearing - math.atan2(float(seen[1]), float(seen[0])))
        error = wrap_angle(-self.turn_target - turned)

        if abs(error) <= self.TURN_TOL or self.back_ticks * self.DT > self.TURN_TIMEOUT:
            self.get_logger().info(
                f"back on the original heading: turned {np.rad2deg(turned):+.1f} deg of "
                f"{np.rad2deg(-self.turn_target):+.1f} deg, {np.rad2deg(error):+.1f} deg out; "
                f"carrying on forwards")
            self.turning_back = False
            self.facing, self.facing_midpoint, self.hug_at = None, None, None
            self.side, self.doorways = None, [] # start the search fresh next time
            self.move_2D(0, 0, 0)
            return

        turn = float(np.clip(self.TURN_GAIN * error, -max_turn_velocity, max_turn_velocity))
        step = -turn * self.DT
        cs, sn = math.cos(step), math.sin(step)
        x, y = float(self.hug_at[0]), float(self.hug_at[1])
        self.hug_at = np.array([x * cs - y * sn, x * sn + y * cs])
        self.get_logger().debug(f"turning back: {np.rad2deg(error):+.1f} deg to go")
        self.move_2D(0, 0, turn)

    def report_doorways(self, doorways, chosen):
        """Print everything known about the openings we are stopping for"""
        log = self.get_logger()
        log.info(f"{len(doorways)} doorway(s) confirmed, stopping")
        for n, doorway in enumerate(sorted(doorways, key=lambda door: door.gap), start=1):
            mark = " <- lined up with this one" if doorway is chosen else ""
            offset, approach, _ = doorway_alignment(doorway)
            log.info(f"  doorway {n}: gap {doorway.gap:.3f} m, centres "
                     f"{doorway.separation:.3f} m apart, bearing "
                     f"{np.rad2deg(doorway.bearing):+.2f} deg, midpoint "
                     f"{doorway.midpoint[0]:.3f} m ahead and {doorway.midpoint[1]:.3f} m left"
                     f"{mark}")
            log.info(f"    off its centre line by {offset:+.3f} m, "
                     f"{approach:.3f} m to run through it")
            for name, can in (("left ", doorway.left), ("right", doorway.right)):
                log.info(f"    {name} can: bearing {np.rad2deg(can.bearing):+.2f} deg, centre "
                         f"{can.distance:.3f} m away, {can.beams} beams, "
                         f"chord {can.chord:.3f} m")

    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return # Does not run if the laser message is not received.

        if self.halted:
            self.move_2D(0, 0, 0) # a doorway has been reported: ramp down and stay put
            return

        if self.turning_back:
            self.turn_back()
            return

        if self.crossing:
            self.cross_doorway()
            return

        if self.facing is not None:
            # Once lined up, finish the turn. It must outrank the forward check, because
            # a doorway the robot is squared up to reads as clear road ahead.
            self.face_doorway()
            return

        # strip ahead is as wide as the robot, strips to the sides are as long as the robot
        front = self.corridor(0, self.ROBOT_WIDTH)
        left = self.corridor(90, self.ROBOT_LEN)
        right = self.corridor(-90, self.ROBOT_LEN)

        # front clear: just go forward
        if front > self.SAFE_DIST:
            self.doorways = []
            self.get_logger().debug(f"forward, front: {front:.2f} m")
            self.move_2D(max_translate_velocity, 0, 0)
            return

        # Only look for doorways while the front is blocked. One straight ahead needs no
        # special case: strafing clear of the obstacle and then driving on goes through it.
        tracked = self.doorway_candidates()
        if self.give_up > 0:
            # An opening was picked and could not be reached. Searching wins for a while,
            # otherwise the same unreachable one is chosen again the moment the robot backs
            # off, and it just rocks between the two.
            self.give_up -= 1
        elif tracked:
            # Only openings the robot could actually drive into. One it is already level
            # with, or has gone past, has nothing to aim at: sliding sideways towards it
            # never ends, because driving forward afterwards would miss it entirely.
            usable = []
            for door, streak in tracked:
                offset, approach, across = doorway_alignment(door)
                if approach >= self.APPROACH_MIN and abs(across[1]) > 0.1:
                    usable.append((door, streak, offset, approach, across))

            if usable:
                # Head for the nearest. Closing before committing is the whole point: the
                # gap reading is good to about 9 mm at 0.4 m and 15 mm at 1 m, against only
                # 35 mm of clearance either side of the robot.
                door, streak, offset, approach, across = min(
                    usable, key=lambda item: float(np.linalg.norm(item[0].midpoint)))

                if (streak >= self.DOORWAY_CONFIRM and abs(offset) <= self.ALIGN_TOL
                        and approach <= self.APPROACH_MAX):
                    # Committed. Work the turn out now, from this opening, and stop looking.
                    normal, _ = doorway_normal(door)
                    self.facing = door
                    self.facing_midpoint = door.midpoint.copy()
                    self.turn_target = math.atan2(float(normal[1]), float(normal[0]))
                    self.turn_done, self.turn_ticks = 0.0, 0
                    self.get_logger().info(
                        f"on the centre line of a doorway: gap {door.gap:.3f} m, off by "
                        f"{offset:+.3f} m, {approach:.3f} m to run; turning "
                        f"{np.rad2deg(self.turn_target):+.1f} deg to face it")
                    self.move_2D(0, 0, 0)
                    return

                # Strafing sideways moves the robot along the chord by the chord's own y
                # component, so this is the direction that shrinks the offset.
                want = 1 if offset * across[1] > 0 else -1
                room = left if want > 0 else right
                if room > self.CLOSE_CLEARANCE:
                    self.side = want
                    speed = min(max_translate_velocity, max(0.08, abs(offset) / 0.4))
                    self.get_logger().debug(
                        f"closing on doorway: gap {door.gap:.3f} m, off centre line by "
                        f"{offset:+.3f} m, {approach:.2f} m to run, strafe "
                        f"{'L' if want > 0 else 'R'} at {speed:.2f} m/s")
                    self.move_2D(0, want * speed, 0)
                    return

                # Cannot get to its centre line from this side. Give up on openings for a
                # moment and go looking the other way instead.
                self.give_up = self.GIVE_UP_TICKS
                self.side = -want
                self.get_logger().info(
                    f"doorway needs {'left' if want > 0 else 'right'} but that side is "
                    f"blocked at {room:.2f} m; searching "
                    f"{'right' if want > 0 else 'left'} for {self.GIVE_UP_TICKS * self.DT:.0f} s")

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

        # Both sides can be under SAFE_DIST at once, and the line above then picks the less
        # bad one. Strafing into it at full speed is how the robot clips a can: it is already
        # committed by the time the next scan lands. Back out instead when there is no room.
        if room <= self.MIN_SIDE_ROOM:
            behind = self.corridor(180, self.ROBOT_WIDTH)
            if behind > self.SAFE_DIST:
                self.get_logger().debug(
                    f"boxed in (left {left:.2f}, right {right:.2f}, front {front:.2f}); "
                    f"backing off with {behind:.2f} m behind")
                self.move_2D(-max_translate_velocity / 2, 0, 0)
            else:
                self.get_logger().debug(
                    f"boxed in with nowhere to go (left {left:.2f}, right {right:.2f}, "
                    f"front {front:.2f}, behind {behind:.2f})")
                self.move_2D(0, 0, 0)
            return

        # Ease off as the gap closes so a late scan cannot leave it committed to a full
        # speed strafe into something.
        speed = min(max_translate_velocity, max(0.08, 2.0 * (room - self.MIN_SIDE_ROOM)))
        self.get_logger().debug(
            f"strafe {'L' if self.side > 0 else 'R'} at {speed:.2f} m/s, room: {room:.2f} m, "
            f"front: {front:.2f} m")
        self.move_2D(0, self.side * speed, 0)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
