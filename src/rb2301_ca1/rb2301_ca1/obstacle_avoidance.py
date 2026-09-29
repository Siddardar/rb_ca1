import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(
    2, suppress=True
)  # Print numpy arrays to specified d.p. and suppress scientific notation (e.g. 1e-5)

set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG


def wrap(a):
    """Fold angle(s) into [-pi, pi), so 350 deg reads as -10 deg rather than a big number"""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


class ObstacleAvoidanceNode(Node):
    """Gap-finding obstacle avoidance for a holonomic base.

    Bearings and yaw are both measured CLOCKWISE FROM THE NOSE, so bearing 0 is the
    nose, +90 is the robot's right, and body coordinates are (x = right, y = nose).
    ROS is counterclockwise-positive and REP-103 fixes the Twist axes, so every
    conversion out of this frame flips a sign -- see move_2D and integrate_yaw.

    Travel direction and nose direction are independent on this base, and the whole
    design turns on keeping them apart. A crossing holds one frozen world direction
    while the nose either points along it (rotating first) or stays where it is.
    """

    CRUISE = "CRUISE"     # no latch: pick a clear direction, unwind residual yaw
    ALIGN = "ALIGN"       # latched, rotating the nose onto the crossing direction
    CROSS = "CROSS"       # latched, driving the crossing direction

    def __init__(self):
        """Node constructor"""
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting Obstacle Avoidance")

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)  # Publish to cmd_vel node
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # The subscriber to the Lidar ranges.

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("robot_width", 0.225)      # across the nose
        self.declare_parameter("robot_length", 0.289)     # along the nose
        self.declare_parameter("clearance", 0.03)         # inflation, per side
        self.declare_parameter("plan_dist", 0.60)         # free distance for full speed
        self.declare_parameter("min_cruise_dist", 0.10)   # useful progress beyond braking room
        self.declare_parameter("heading_step_deg", 3.0)   # cruise direction grid
        self.declare_parameter("hysteresis", 0.5)         # must be < 1, see choose()
        self.declare_parameter("room_gain", 0.25)         # room_gain*horizon must stay < pi
        self.declare_parameter("horizon", 2.0)            # free distance past this is all equal
        self.declare_parameter("yaw_gain", 1.5)           # unwind / align P gain
        self.declare_parameter("align_tol_deg", 4.0)      # when ALIGN is done
        self.declare_parameter("max_gap_span_deg", 120.0) # widest edge pair to consider
        self.declare_parameter("motion_samples", 8)       # trajectory samples per check
        self.declare_parameter("cross_timeout", 200)      # ticks a latch may live
        self.declare_parameter("max_speed", 0.4)
        self.declare_parameter("max_turn", 0.8)
        # Bearing of scan index 0, measured clockwise from the nose. Set it by putting
        # one object dead ahead and checking the scan minimum lands near bearing 0.
        self.declare_parameter("lidar_mount_yaw", math.pi)

        g = lambda n: self.get_parameter(n).value
        self.W = g("robot_width")
        self.L = g("robot_length")
        self.c = g("clearance")
        self.d_plan = g("plan_dist")
        self.min_cruise_dist = g("min_cruise_dist")
        self.hyst = g("hysteresis")
        self.room_gain = g("room_gain")
        self.horizon = g("horizon")
        self.yaw_gain = g("yaw_gain")
        self.align_tol = math.radians(g("align_tol_deg"))
        self.max_span = math.radians(g("max_gap_span_deg"))
        self.n_samples = int(g("motion_samples"))
        self.cross_timeout = int(g("cross_timeout"))
        self.max_v = g("max_speed")
        self.max_w = g("max_turn")
        self.lidar_yaw = g("lidar_mount_yaw")

        # Footprint as an axis-aligned box in body coordinates: half-extents across
        # the nose and along it. Inflated for collision, raw for the overlap monitor.
        self.ex, self.ey = self.W / 2 + self.c, self.L / 2 + self.c
        self.ex_raw, self.ey_raw = self.W / 2, self.L / 2

        # ---- cruise direction grid ---------------------------------------
        step = math.radians(g("heading_step_deg"))
        self.grid = wrap(np.arange(int(round(2 * math.pi / step))) * step)
        self.grid_ux = np.sin(self.grid)    # body x (right) component
        self.grid_uy = np.cos(self.grid)    # body y (nose)  component

        # ---- scan cache ---------------------------------------------------
        # Beam bearings are fixed for a given sensor, so the trig is computed once.
        # Keyed on the full scan geometry: a sensor can report the same sample count
        # at a different increment, and length alone would miss that.
        self._key = None
        self._bearings = self._sin_b = self._cos_b = None
        self._scan = self.px = self.py = None

        # ---- controller state --------------------------------------------
        self.DT = 0.05 # controller period (s)
        self.timer = self.create_timer(self.DT, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.MAX_ACCEL = 0.8 # m/s^2 per axis; lower = gentler direction changes
        self.MAX_ALPHA = 4.0 # rad/s^2; limit angular acceleration for physical tracking
        self.cmd_x = 0.0 # velocity actually sent last tick, so we can ramp towards the new target
        self.cmd_y = 0.0
        self.cmd_turn = 0.0

        self.yaw = 0.0          # psi: rotation since startup, CLOCKWISE-positive
        self.t_prev = None
        self.world_prev = 0.0   # last tick's chosen direction, startup-relative
        self.state = self.CRUISE

        self.theta_star = 0.0   # latched crossing direction, startup-relative
        self.edges = None       # latched edge points, body coords, shape (2, 2)
        self.needs_rotation = False
        self.latch_ticks = 0
        self.block = 0          # ticks before another latch is allowed

    # ---------------------------------------------------------------- output

    def regularise(self, target, current, rate):
        """Move current towards target by at most rate * DT, so commands ramp instead of jumping"""
        step = rate * self.DT
        return float(np.clip(target, current - step, current + step))

    def move_2D(self, beta: float, v: float, turn: float):
        """Publish a twist travelling at speed v along body bearing beta, turning at `turn`.

        beta is clockwise from the nose, so the nose component is cos and the LEFT
        component is -sin. turn is angular.z, i.e. counterclockwise-positive, which is
        the opposite sense to self.yaw. Check the rate-limited command and its braking
        path before publishing; yaw and edge tracking integrate only that final command.
        """
        x = np.clip(v * math.cos(beta), -self.max_v, self.max_v)
        y = np.clip(-v * math.sin(beta), -self.max_v, self.max_v)
        turn = np.clip(turn, -self.max_w, self.max_w)

        x = self.regularise(x, self.cmd_x, self.MAX_ACCEL)
        y = self.regularise(y, self.cmd_y, self.MAX_ACCEL)
        turn = self.regularise(turn, self.cmd_turn, self.MAX_ALPHA)
        if not self.command_clear(x, y, turn):
            # Prefer a checked, controlled stop. If even braking would contact an
            # obstacle, command an immediate stop rather than publishing unsafe motion.
            x = self.regularise(0.0, self.cmd_x, self.MAX_ACCEL)
            y = self.regularise(0.0, self.cmd_y, self.MAX_ACCEL)
            turn = self.regularise(0.0, self.cmd_turn, self.MAX_ALPHA)
            if not self.command_clear(x, y, turn):
                x, y, turn = 0.0, 0.0, 0.0
        self.cmd_x, self.cmd_y, self.cmd_turn = x, y, turn

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = self.cmd_x, self.cmd_y, 0.0
        msg.angular.x, msg.angular.y, msg.angular.z = 0.0, 0.0, self.cmd_turn
        self.pub_cmd_vel.publish(msg)

    # ----------------------------------------------------------------- input

    def sub_scan_callback(self, msg):
        """Scan subscriber. Converts to body-frame points once per scan."""
        scan = np.asarray(msg.ranges, dtype=float) # keep full scan

        bad = ~np.isfinite(scan) | (scan < msg.range_min) # clean up values
        scan[bad] = msg.range_max

        key = (len(scan), msg.angle_min, msg.angle_increment)
        if key != self._key:
            # The sensor counts counterclockwise in its own frame; negate to get a
            # clockwise-from-nose bearing, and add the mount offset.
            self._bearings = wrap(
                -(msg.angle_min + np.arange(len(scan)) * msg.angle_increment + self.lidar_yaw))
            self._sin_b, self._cos_b = np.sin(self._bearings), np.cos(self._bearings)
            self._key = key

        self._scan = scan
        self.px = scan * self._sin_b     # body x: right
        self.py = scan * self._cos_b     # body y: nose

    # ------------------------------------------------- collision  (Part 3.1)

    def _slab(self, p, u, e):
        """t-interval over which p - t*u stays within [-e, e] on one axis."""
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (p - e) / u, (p + e) / u
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        parallel = np.abs(u) < 1e-12        # ray never crosses this slab boundary
        within = np.abs(p) <= e
        lo = np.where(parallel, np.where(within, -np.inf, np.inf), lo)
        hi = np.where(parallel, np.where(within, np.inf, -np.inf), hi)
        return lo, hi

    def _box_distances(self, px, py, ux, uy, ex, ey):
        """Entry distance for every point/direction pair, including initial overlap."""
        lo_x, hi_x = self._slab(px[:, None], np.atleast_1d(ux)[None, :], ex)
        lo_y, hi_y = self._slab(py[:, None], np.atleast_1d(uy)[None, :], ey)
        t_in, t_out = np.maximum(lo_x, lo_y), np.minimum(hi_x, hi_y)
        hit = (t_in <= t_out) & (t_out >= 0.0)
        return np.where(hit, np.maximum(t_in, 0.0), np.inf)

    def _sweep_points(self, px, py, ux, uy):
        """Plan with inflation, retaining directional clearance inside the margin."""
        distances = self._box_distances(px, py, ux, uy, self.ex, self.ey)
        in_margin = (np.abs(px) <= self.ex) & (np.abs(py) <= self.ey)
        if np.any(in_margin):
            # A nearby side obstacle may permit retreat or parallel movement, but
            # must still constrain movement toward the actual body. Never discard it.
            raw = self._box_distances(px[in_margin], py[in_margin], ux, uy,
                                      self.ex_raw, self.ey_raw)
            distances[in_margin] = np.maximum(raw - self.c, 0.0)
        return distances.min(axis=0)

    def sweep(self, ux, uy):
        """Planned free travel using exact box intersections and a clearance margin.

        In body coordinates the footprint is an AXIS-ALIGNED box, so nothing rotates --
        the travel direction tilts, the box does not. Free travel is then a two-slab
        ray-box intersection per beam, solving for p - t*u inside the box. Exact, and it
        returns a distance rather than a boolean, which is what speed control needs.

        A strip test on the bounding box of the rotated footprint would be up to 2x too
        large at diagonal travel, refusing motion the body would clear.
        """
        return self._sweep_points(self.px, self.py, ux, uy)

    def sweep_one(self, beta):
        """Free travel along a single body bearing."""
        return float(self.sweep(math.sin(beta), math.cos(beta))[0])

    def motion_clear(self, dx, dy, dpsi):
        """Is translating by (dx, dy) while rotating dpsi collision-free?

        Checking a translation and a rotation separately does not establish that doing
        both is clear: the corners follow different paths when they happen together. So
        sample the motion and test box membership at each sample. Pure rotation is the
        dx = dy = 0 case, and it covers the swept union over the interval, which is
        strictly larger than either endpoint because the corners trace arcs.
        """
        s = np.linspace(0.0, 1.0, max(2, self.n_samples))
        return self._poses_clear(dx * s, dy * s, dpsi * s)

    def _poses_clear(self, dx, dy, dpsi):
        """Check poses expressed relative to the current body frame."""
        if self.px is None:
            return False
        X = self.px[:, None] - np.asarray(dx)[None, :]
        Y = self.py[:, None] - np.asarray(dy)[None, :]
        cs, sn = np.cos(dpsi), np.sin(dpsi)
        # Frame turns clockwise, so fixed points rotate by the opposite angle.
        Xr = X * cs[None, :] - Y * sn[None, :]
        Yr = Y * cs[None, :] + X * sn[None, :]
        # true footprint, not the inflated one: sweep() already plans with margin, and
        # this only has to stop the robot short of contact
        return not bool(np.any((np.abs(Xr) < self.ex_raw) & (np.abs(Yr) < self.ey_raw)))

    def command_clear(self, x, y, turn):
        """Check one command period followed by braking at the configured slew rates.

        Integrate body-frame velocities along the turning trajectory. This checks the
        command after rate limiting, including residual motion when the target is zero.
        It is a command-based prediction, not measured stopping performance.
        """
        dx = dy = psi = 0.0
        poses = [(dx, dy, psi)]
        samples = max(2, self.n_samples)
        interval = self.DT / samples
        while max(abs(x), abs(y), abs(turn)) > 1e-12:
            vx, vy, omega = -y, x, -turn  # right, nose, clockwise
            angle = omega * interval
            if abs(omega) < 1e-12:
                local_x, local_y = vx * interval, vy * interval
            else:
                a = math.sin(angle) / omega
                b = (1.0 - math.cos(angle)) / omega
                local_x, local_y = a * vx + b * vy, -b * vx + a * vy
            for _ in range(samples):
                cs, sn = math.cos(psi), math.sin(psi)
                dx += cs * local_x + sn * local_y
                dy += -sn * local_x + cs * local_y
                psi += angle
                poses.append((dx, dy, psi))
            x = self.regularise(0.0, x, self.MAX_ACCEL)
            y = self.regularise(0.0, y, self.MAX_ACCEL)
            turn = self.regularise(0.0, turn, self.MAX_ALPHA)
        poses = np.asarray(poses)
        return self._poses_clear(poses[:, 0], poses[:, 1], poses[:, 2])

    def overlapping(self):
        """Anything inside the true footprint. Should never be true; last line of defence."""
        return bool(np.any((np.abs(self.px) < self.ex_raw) & (np.abs(self.py) < self.ey_raw)))

    # ----------------------------------------------------- footprint extents

    def w_of(self, delta):
        """Corridor width swept translating along delta (delta = travel-vs-nose angle)."""
        return self.W * abs(math.cos(delta)) + self.L * abs(math.sin(delta))

    def h_of(self, delta):
        """Reach from the centre to the leading (and trailing) edge along delta."""
        return 0.5 * (self.L * abs(math.cos(delta)) + self.W * abs(math.sin(delta)))

    # ------------------------------------------------- gap finding  (Part 1)

    def find_gaps(self):
        """Candidate crossings, as (score, beta, needs_rotation, Pa, Pb)."""
        r = self._scan
        n = len(r)
        jump = np.roll(r, -1) - r                      # J_i = r_{i+1} - r_i
        thr = self.W / 2                               # smaller steps cannot hide a passage
        rising = np.flatnonzero(jump > thr)             # range rises: near point is beam i
        falling = np.flatnonzero(jump < -thr)           # range falls: near point is beam i+1
        if rising.size == 0 or falling.size == 0:
            return []

        span_cap = self.max_span / abs(self._bearings[1] - self._bearings[0])
        out = []
        for p in rising:
            k = int(np.searchsorted(falling, p, side="right")) % falling.size
            q = int(falling[k])
            span = (q - p) % n
            if span < 1 or span > span_cap:
                continue
            ia, ib = int(p), int((q + 1) % n)
            # the beams between must see past both edges, or this is not an opening
            through = np.take(r, np.arange(p + 1, p + 1 + span), mode="wrap")
            if through.min() <= max(r[ia], r[ib]):
                continue
            cand = self.assess(ia, ib)
            if cand is not None:
                out.append(cand)
        return out

    def assess(self, ia, ib):
        """Steps 4-6 for one edge pair. None if it is not crossable."""
        Pa = np.array([self.px[ia], self.py[ia]])
        Pb = np.array([self.px[ib], self.py[ib]])
        chord = Pb - Pa
        g = float(np.hypot(*chord))
        if g <= self.W + 2 * self.c:
            return None                                # too narrow at any orientation

        M = 0.5 * (Pa + Pb)
        beta = math.atan2(M[0], M[1])                  # bearing is atan2(x, y) in this frame
        # The opening only counts across the path, so project the chord perpendicular to
        # travel. This is what makes the gap normal unnecessary: obliquity enters here.
        n_u = np.array([-math.cos(beta), math.sin(beta)])
        opening = abs(float(chord @ n_u))

        # w() is minimised at delta = 0, so facing the travel direction is the cheapest
        # orientation; not rotating means delta = beta.
        if self.w_of(beta) + 2 * self.c < opening:
            needs_rotation = False
        elif self.W + 2 * self.c < opening:
            needs_rotation = True
        else:
            return None

        # Rotation-required crossings align in place first. Assess translation in
        # that resulting frame, not with the unrotated footprint that cannot fit.
        if needs_rotation:
            if not self.motion_clear(0.0, 0.0, beta):
                return None
            cs, sn = math.cos(beta), math.sin(beta)
            px = self.px * cs - self.py * sn
            py = self.py * cs + self.px * sn
            free = float(self._sweep_points(px, py, 0.0, 1.0)[0])
        else:
            free = self.sweep_one(beta)
        if free < float(np.hypot(*M)):
            return None

        score = abs(float(wrap(beta + self.yaw)))       # deviation from the startup heading
        return score, beta, needs_rotation, Pa, Pb

    # --------------------------------------------------- cruise  (Part 2, 4)

    def choose(self):
        """Body bearing of the best clear direction, or None.

        Free distance is a hard constraint. It returns as a bonus only saturated at the
        horizon: an unbounded room reward makes reversing win whenever the way back is
        more open than the way forward, which is always true when something blocks you.
        With room_gain * horizon < pi that can never outweigh turning around.

        Hysteresis is measured in the STARTUP frame. Penalising the body-frame change
        would charge the robot for tracking a straight world line while it rotates.
        """
        free = self.sweep(self.grid_ux, self.grid_uy)
        speed = math.hypot(self.cmd_x, self.cmd_y)
        braking_room = speed * self.DT + speed * speed / (2 * self.MAX_ACCEL)
        feasible = free > self.min_cruise_dist + braking_room
        if not feasible.any():
            return None, free
        world = wrap(self.grid + self.yaw)              # goal is startup bearing 0
        cost = (np.abs(world)
                + self.hyst * np.abs(wrap(world - self.world_prev))
                - self.room_gain * np.minimum(free, self.horizon))
        return int(np.argmin(np.where(feasible, cost, np.inf))), free

    def cruise(self):
        """No latch: drive the clearest direction nearest the startup heading."""
        if self.block:
            self.block -= 1
        else:
            gaps = self.find_gaps()
            if gaps:
                return self.latch(min(gaps))

        i, free = self.choose()
        if i is None:
            self.get_logger().warn("no clear direction")
            return 0.0, 0.0, 0.0

        beta, room = float(self.grid[i]), float(free[i])
        v = self.max_v * min(1.0, room / self.d_plan)
        # The cost minimum already sits at beta = -yaw, so translating there holds a
        # straight world line while this term rotates the body upright underneath it.
        # angular.z is counterclockwise, yaw is clockwise, hence the plus sign.
        turn = float(np.clip(self.yaw_gain * self.yaw, -self.max_w, self.max_w))
        if not self.motion_clear(v * math.sin(beta) * self.DT,
                                 v * math.cos(beta) * self.DT, -turn * self.DT):
            turn = 0.0
            if not self.motion_clear(v * math.sin(beta) * self.DT,
                                     v * math.cos(beta) * self.DT, 0.0):
                v = 0.0
        self.world_prev = float(wrap(beta + self.yaw))
        return beta, v, turn

    # ----------------------------------------------------- crossing (Part 2)

    def latch(self, cand):
        """Commit to a crossing. The direction is frozen from here, never recomputed."""
        _, beta, needs_rotation, Pa, Pb = cand
        self.theta_star = float(wrap(beta + self.yaw))   # startup-relative, so it survives
        self.edges = np.stack([Pa, Pb])
        self.needs_rotation = needs_rotation
        self.latch_ticks = 0
        self.state = self.ALIGN if needs_rotation else self.CROSS
        self.get_logger().info(
            f"{self.state} bearing {math.degrees(beta):+.0f} deg, "
            f"{'rotating' if needs_rotation else 'strafing'}")
        return None

    def track_edges(self, dt):
        """Carry the latched edge points through the motion just commanded.

        Detection cannot do this job. As the robot enters the opening its two edges
        spread toward opposite sides of the scan and stop satisfying the pairing rules,
        which is what a SUCCESSFUL crossing looks like -- so losing the candidate must
        never release the latch. A crossing lasts under a second, so this integration is
        bounded by construction.
        """
        dx, dy = -self.cmd_y * dt, self.cmd_x * dt      # body (right, nose) displacement
        dpsi = -self.cmd_turn * dt                      # angular.z is CCW, psi is CW
        p = self.edges - np.array([dx, dy])
        cs, sn = math.cos(dpsi), math.sin(dpsi)
        self.edges = np.stack([p[:, 0] * cs - p[:, 1] * sn,
                               p[:, 1] * cs + p[:, 0] * sn], axis=1)

    def crossing(self):
        """ALIGN and CROSS. No re-selection, no heading recovery."""
        delta = float(wrap(self.theta_star - self.yaw))
        u = np.array([math.sin(delta), math.cos(delta)])
        self.latch_ticks += 1
        if self.latch_ticks > self.cross_timeout:
            return self.release("timed out", block=40)
        # Cleared once BOTH edges are behind the trailing edge. Computed from the
        # tracked points, so it holds even after the detector has lost the gap.
        f = self.edges @ u
        if bool(np.all(f < -self.h_of(delta))):
            return self.release("cleared", block=0)

        if self.needs_rotation and abs(delta) > self.align_tol:
            self.state = self.ALIGN
            # psi must grow toward theta_star, i.e. rotate clockwise for delta > 0,
            # which is negative angular.z.
            turn = float(np.clip(-self.yaw_gain * delta, -self.max_w, self.max_w))
            # Match assess(): finish alignment before starting the approach. Final
            # output validation also checks any residual translation while braking.
            if not self.motion_clear(0.0, 0.0, delta):
                return self.release("no room to align", block=40)
            return delta, 0.0, turn

        self.state = self.CROSS
        free = self.sweep_one(delta)
        if free <= 0.0:
            return self.release("blocked", block=40)
        v = self.max_v * min(1.0, free / self.d_plan)
        if not self.motion_clear(v * u[0] * self.DT, v * u[1] * self.DT, 0.0):
            return self.release("would contact while crossing", block=40)
        return delta, v, 0.0

    def release(self, why, block):
        """Drop the latch and hand back to cruise."""
        self.get_logger().info(f"{self.state} released: {why}")
        self.state, self.edges, self.block = self.CRUISE, None, block
        return None

    # ------------------------------------------------------------------ loop

    def timer_callback(self):
        """Controller loop"""

        if self.px is None:
            return # Does not run if the laser message is not received.

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = self.DT if self.t_prev is None else min(max(now - self.t_prev, 0.0), 5 * self.DT)
        self.t_prev = now

        # Dead reckoning: integrate what was actually published, after rate limiting.
        # angular.z is counterclockwise-positive and psi is clockwise-positive.
        self.yaw = float(wrap(self.yaw - self.cmd_turn * dt))
        if self.edges is not None:
            self.track_edges(dt)

        # A handler returns None when it changed state, so the new state acts this tick.
        cmd = None
        for _ in range(3):
            cmd = self.cruise() if self.state == self.CRUISE else self.crossing()
            if cmd is not None:
                break
        if cmd is None:
            cmd = (0.0, 0.0, 0.0)

        beta, v, turn = cmd
        if self.overlapping():
            self.get_logger().warn("something is inside the footprint, stopping")
            v, turn = 0.0, 0.0
        self.move_2D(beta, v, turn)


def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
