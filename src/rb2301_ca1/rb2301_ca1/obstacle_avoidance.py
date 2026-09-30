"""Reactive gap avoidance: search with a turning circle, cross with a rectangle."""

import math
from collections import Counter

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def direction(beta):
    """Body coordinates: x right, y forward; bearings clockwise from the nose."""
    return np.array([math.sin(beta), math.cos(beta)])


class ObstacleAvoidanceNode(Node):
    FORWARD = "FORWARD"
    SEARCH = "SEARCH"
    ADJUST = "ADJUST"
    TURN = "TURN"
    CROSS = "CROSS"
    EXIT = "EXIT"  # gap cleared; hold heading until there is room to restore it
    RECOVER = "RECOVER"
    RETREAT = "RETREAT"

    def __init__(self):
        super().__init__("obstacle_avoidance")
        defaults = {
            "robot_width": 0.225, "robot_length": 0.289, "clearance": 0.03,
            "max_speed": 0.4, "reposition_speed": 0.15, "max_turn": 0.8,
            "max_accel": 0.8, "max_angular_accel": 4.0, "yaw_gain": 1.5,
            "align_tol_deg": 0.5, "cross_stop_dist": 0.05,
            "forward_resume_dist": 0.35, "reposition_tolerance": 0.02,
            "reposition_max_dist": 0.60, "min_gap_slack": 0.01,
            "max_gap_span_deg": 180.0, "gap_max_distance": 1.5,
            "progress_timeout": 2.0, "progress_dist": 0.01,
            "failed_gap_timeout": 5.0, "failed_gap_match_dist": 0.15,
            "lidar_mount_yaw": math.pi, "lidar_offset_forward": 0.11423,
            "lidar_offset_right": 0.0, "debug_logging": True,
            "debug_interval": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        g = lambda name: self.get_parameter(name).value
        self.W, self.L, self.margin = g("robot_width"), g("robot_length"), g("clearance")
        self.radius = math.hypot(self.W / 2, self.L / 2) + self.margin
        self.half_width = self.W / 2 + self.margin
        self.half_length = self.L / 2 + self.margin
        self.max_v, self.side_v, self.max_w = g("max_speed"), g("reposition_speed"), g("max_turn")
        self.accel, self.alpha = g("max_accel"), g("max_angular_accel")
        self.yaw_gain = g("yaw_gain")
        self.align_tol = math.radians(g("align_tol_deg"))
        self.stop = g("cross_stop_dist")
        self.resume = max(g("forward_resume_dist"), self.stop + 0.05)
        self.position_tol = g("reposition_tolerance")
        self.adjust_max = g("reposition_max_dist")
        self.min_slack = g("min_gap_slack")
        self.max_span = math.radians(min(180.0, g("max_gap_span_deg")))
        self.gap_range = g("gap_max_distance")
        self.progress_timeout, self.progress_dist = g("progress_timeout"), g("progress_dist")
        self.failed_timeout, self.failed_match = g("failed_gap_timeout"), g("failed_gap_match_dist")
        self.lidar_yaw = g("lidar_mount_yaw")
        self.lidar_offset = np.array([g("lidar_offset_right"), g("lidar_offset_forward")])
        self.debug_enabled, self.debug_interval = g("debug_logging"), g("debug_interval")
        if self.debug_enabled:
            self.get_logger().set_level(LoggingSeverity.DEBUG)

        self.DT = 0.05
        self.points = None
        self.beam_points = None
        self.ranges = self.valid = None
        self.angle_step = 0.0
        self.scan_time = self.cloud_time = None
        self.last_tick = None
        self.now = 0.0
        self.yaw = 0.0
        self.velocity = np.zeros(2)  # last published (right, forward) translation
        self.omega = 0.0             # last published CLOCKWISE rotation
        self.state = self.FORWARD
        self.side = 1                # +1 right, -1 left; held throughout a search leg
        self.last_side_switch = -math.inf
        self.backing = False         # SEARCH is reversing out of a boxed-in position
        self.lidar_origin = self.lidar_offset.copy()
        self.edges = self.waypoint = self.entry = None
        self.cross_heading = 0.0     # frozen bearing relative to startup
        self.turn_ready = False
        self.plan_reason = ""
        self.failed = []             # (expiry, motion-tracked endpoint pair)
        self.progress_key = None
        self.progress_value = self.progress_time = 0.0
        self.debug_times = {}
        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.scan_callback, 2)
        self.timer = self.create_timer(self.DT, self.timer_callback)
        self.get_logger().info(
            f"Starting Obstacle Avoidance: bubble_radius={self.radius:.3f}m "
            f"crossing_width={2 * self.half_width:.3f}m; continuous left/right search; "
            f"lidar_offset={self.lidar_offset.tolist()} mount_yaw={math.degrees(self.lidar_yaw):.1f}deg")

    def debug(self, key, message):
        if self.debug_enabled and self.now - self.debug_times.get(key, -math.inf) >= self.debug_interval:
            self.debug_times[key] = self.now
            self.get_logger().debug(message)

    def change_state(self, state, reason):
        self.get_logger().info(f"{self.state} -> {state}: {reason}")
        self.state = state
        self.progress_key = None
        self.backing = False

    def scan_callback(self, msg):
        raw = np.asarray(msg.ranges, dtype=float)
        # Returns closer than range_min are still real obstacles (just less
        # accurate); treating them as no-return would open a phantom gap ahead.
        self.valid = np.isfinite(raw) & (raw > 0.0) & (raw < msg.range_max)
        self.ranges = np.where(self.valid, raw, msg.range_max)
        bearings = -(msg.angle_min + np.arange(len(raw)) * msg.angle_increment + self.lidar_yaw)
        self.beam_points = np.column_stack((self.ranges * np.sin(bearings),
                                           self.ranges * np.cos(bearings))) + self.lidar_offset
        # No-return beams are not obstacle endpoints or synthetic distant obstacles.
        self.points = self.beam_points[self.valid].copy()
        self.lidar_origin = self.lidar_offset.copy()
        self.angle_step = abs(msg.angle_increment)
        self.scan_time = self.cloud_time = self.get_clock().now().nanoseconds * 1e-9
        self.debug("scan", f"SCAN beams={len(raw)} returns={np.count_nonzero(self.valid)} "
                   f"no_return_or_invalid={np.count_nonzero(~self.valid)}")

    # Search geometry: one circle guarantees room for any in-place turn.
    def bubble_clear(self, origin=None):
        p = self.points if origin is None else self.points - origin
        return not np.any(np.linalg.norm(p, axis=1) < self.radius - 1e-6)

    def bubble_free(self, beta, origin=None):
        p = self.points if origin is None else self.points - origin
        if not len(p):
            return math.inf
        u = direction(beta)
        along = p @ u
        perpendicular2 = np.maximum(np.sum(p * p, axis=1) - along * along, 0.0)
        hits = (perpendicular2 < self.radius ** 2) & (along >= 0.0)
        # A point already inside the circle only blocks directions that would
        # carry the circle deeper onto it; moving away from it is the escape.
        inside = np.linalg.norm(p, axis=1) < self.radius - 1e-6
        if np.any(inside & (along > 0.0)):
            return 0.0
        hits &= ~inside
        if not np.any(hits):
            return math.inf
        return max(0.0, float(np.min(along[hits] - np.sqrt(self.radius ** 2 - perpendicular2[hits]))))

    # Crossing geometry: the actual rectangular footprint, never the search circle.
    def rectangle_free(self, beta, origin=None, heading=0.0):
        p = self.points if origin is None else self.points - origin
        cs, sn = math.cos(heading), math.sin(heading)
        p = np.column_stack((p[:, 0] * cs - p[:, 1] * sn,
                             p[:, 1] * cs + p[:, 0] * sn))
        u = direction(beta - heading)
        entry = np.full(len(p), -np.inf)
        leave = np.full(len(p), np.inf)
        for axis, extent in ((0, self.half_width), (1, self.half_length)):
            if abs(u[axis]) < 1e-10:
                outside = np.abs(p[:, axis]) > extent
                entry[outside], leave[outside] = np.inf, -np.inf
            else:
                a, b = (p[:, axis] - extent) / u[axis], (p[:, axis] + extent) / u[axis]
                entry = np.maximum(entry, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        hits = (entry <= leave) & (leave >= 0.0)
        return max(0.0, float(np.min(entry[hits]))) if np.any(hits) else math.inf

    # Preserve the range-discontinuity / two-obstacle-edge gap finder.
    def edge_pairs(self):
        """Beam pairs (a, b): a is the last return before an opening, b the first after it."""
        n = len(self.ranges)
        counts, pairs = Counter(), []
        if n < 2 or self.angle_step <= 0.0:
            return pairs, counts
        jump = np.roll(self.ranges, -1) - self.ranges
        near = self.valid & (np.linalg.norm(self.beam_points, axis=1) <= self.gap_range)
        rising = np.flatnonzero((jump > self.W / 2) & near)
        falling = np.flatnonzero((jump < -self.W / 2) & np.roll(near, -1))
        counts["rising"], counts["falling"] = len(rising), len(falling)
        for a in rising:
            if not len(falling):
                break
            k = int(np.searchsorted(falling, a, side="right")) % len(falling)
            q = int(falling[k])
            span = (q - int(a)) % n
            if span < 1 or (span + 1) * self.angle_step > self.max_span + 1e-9:
                counts["span"] += 1
                continue
            pairs.append((int(a), (q + 1) % n))
        return pairs, counts

    def edge_bound(self, i, j, toward):
        """Innermost position of the obstacle edge whose last return is beam i.

        The true edge lies between the return of beam i and the ray of the
        neighbouring beam j that passes the obstacle. Extending the obstacle
        surface (estimated from the return next to i, else the chord) to that
        ray bounds the opening conservatively instead of overstating it by up
        to one beam step per side, which biased the crossing midpoint.
        """
        n = len(self.beam_points)
        p, origin = self.beam_points[i], self.lidar_origin
        ray = self.beam_points[j] - origin
        cap = self.W / 2
        k = (2 * i - j) % n
        surface = p - self.beam_points[k]
        guesses = [toward]
        if self.valid[k] and 1e-6 < np.linalg.norm(surface) < cap:
            guesses.insert(0, surface)
        d = origin - p
        for w in guesses:
            det = ray[0] * w[1] - ray[1] * w[0]
            if abs(det) < 1e-9:
                continue
            s = (ray[0] * d[1] - ray[1] * d[0]) / det
            t = (w[0] * d[1] - w[1] * d[0]) / det
            shift = s * w
            if t > 0.0 and shift @ toward > 0.0 and np.linalg.norm(shift) <= cap:
                return p + shift
        nominal = min(cap, float(np.linalg.norm(p - origin)) * self.angle_step)
        return p + toward * nominal

    def pair_edges(self, a, b):
        """Return (estimated, measured) edge pairs for one opening.

        `measured` is where the beams actually returned. Each true edge lies
        between that return and the innermost position the neighbouring beam
        allows, so the measured pair overstates the opening by up to one beam
        step per side and biases its midpoint whenever the two sides are seen at
        different incidence. `estimated` takes the middle of each edge's bracket,
        which is unbiased in both width and centre; crossing geometry uses it,
        and the optimistic `measured` pair only gates whether to approach.
        """
        n = len(self.beam_points)
        raw = self.beam_points[[a, b]].copy()
        chord = raw[1] - raw[0]
        width = float(np.linalg.norm(chord))
        if width < 1e-6:
            return raw, raw
        toward = chord / width
        bounded = np.array([self.edge_bound(a, (a + 1) % n, toward),
                            self.edge_bound(b, (b - 1) % n, -toward)])
        return (bounded + raw) / 2, raw

    def refresh_edges(self):
        """Replace the dead-reckoned edge pair with its match in the current scan."""
        best, best_error = None, self.failed_match
        for a, b in self.edge_pairs()[0]:
            edges = self.pair_edges(a, b)[0]
            error = min(float(np.max(np.linalg.norm(edges - self.edges, axis=1))),
                        float(np.max(np.linalg.norm(edges[::-1] - self.edges, axis=1))))
            if error < best_error:
                best, best_error = edges, error
        if best is None:
            self.get_logger().info("gap not re-detected in current scan; keeping tracked edges")
            return False
        self.get_logger().info(f"gap re-detected: tracked={self.edges.round(3).tolist()} "
                               f"fresh={best.round(3).tolist()} shift={best_error:.3f}m")
        self.edges = best
        return True

    def find_gaps(self):
        pairs, counts = self.edge_pairs()
        candidates = []
        for a, b in pairs:
            edges, measured = self.pair_edges(a, b)
            midpoint = edges.mean(axis=0)
            beta = math.atan2(midpoint[0], midpoint[1])
            world = wrap(beta + self.yaw)
            # The search leg's direction must not veto a reachable gap: a usable
            # opening on the far side is still progress, and refusing it sends the
            # robot strafing past the obstacle instead of through it.
            if math.cos(world) <= 0.0:
                counts["rearward"] += 1
                continue
            if any(min(np.max(np.linalg.norm(edges - old, axis=1)),
                       np.max(np.linalg.norm(edges[::-1] - old, axis=1))) < self.failed_match
                   for _, old in self.failed):
                counts["recent_failure"] += 1
                continue
            chord = edges[1] - edges[0]
            width = float(np.linalg.norm(chord))
            if float(np.linalg.norm(measured[1] - measured[0])) < 2 * self.half_width + self.min_slack:
                counts["width"] += 1
                continue
            if width < 1e-6:
                counts["width"] += 1
                continue
            # Compare crossing from here with one simple staging point on the
            # perpendicular through the gap centre; no sampled pose optimisation.
            normal = np.array([-chord[1], chord[0]]) / width
            if normal @ midpoint < 0.0:
                normal = -normal
            stand_off = max(float(midpoint @ normal), self.radius + self.stop)
            staging = midpoint - stand_off * normal
            accepted = None
            for origin in (np.zeros(2), staging):
                distance = float(np.linalg.norm(origin))
                if distance > self.adjust_max:
                    counts["adjustment_too_far"] += 1
                    continue
                if distance > self.position_tol:
                    adjustment = math.atan2(origin[0], origin[1])
                    if self.bubble_free(adjustment) <= distance + self.stop:
                        counts["adjustment_circle_blocked"] += 1
                        continue
                plan = self.gap_plan(edges, origin, width_edges=measured)
                if plan is not None:
                    plan.update(edges=edges, origin=origin, beams=(a, b))
                    if accepted is None or plan["slack"] > accepted["slack"] + 1e-6:
                        accepted = plan
                else:
                    counts[self.plan_reason] += 1
            if accepted is None:
                counts["approach_or_crossing"] += 1
            else:
                candidates.append(accepted)
                counts["accepted"] += 1
        self.debug("gaps", f"GAPS side={'right' if self.side > 0 else 'left'} outcomes={dict(counts)}")
        return candidates

    def gap_plan(self, edges, origin, width_edges=None):
        """Check gap width, turning room here, and clearance to begin crossing.

        `width_edges` supplies the pair used for the fit test. Deciding to
        approach may use the measured pair, because the bound tightens as the
        robot closes in; committing to cross must leave it None so the test runs
        on the conservative pair, which cannot overstate the opening.
        """
        if not self.bubble_clear(origin):
            self.plan_reason = "turning_circle_blocked"
            return None
        shifted = edges - origin
        midpoint = shifted.mean(axis=0)
        beta = math.atan2(midpoint[0], midpoint[1])
        if math.cos(wrap(beta + self.yaw)) <= 0.0:
            self.plan_reason = "crossing_rearward"
            return None
        u = direction(beta)
        across = np.array([u[1], -u[0]])
        opening = abs(float((edges[1] - edges[0]) @ across))
        slack = opening - 2 * self.half_width
        fit = slack if width_edges is None else \
            abs(float((width_edges[1] - width_edges[0]) @ across)) - 2 * self.half_width
        if fit < self.min_slack:
            self.plan_reason = "projected_width"
            return None
        # This distance is an estimate for logging, not a required clear horizon.
        # Unknown/blocked space farther along the route does not veto entering a
        # usable gap. Actual motion is checked and speed-limited on every tick.
        travel = float(np.max(shifted @ u)) + self.half_length
        free = self.rectangle_free(beta, origin, heading=beta)
        if free <= self.stop + 0.005:
            self.plan_reason = "crossing_entry_blocked"
            return None
        return dict(beta=beta, opening=opening, slack=slack, travel=travel, free=free)

    def latch_gap(self, plan):
        self.edges = plan["edges"].copy()
        self.waypoint = plan["origin"].copy()
        self.cross_heading = wrap(plan["beta"] + self.yaw)
        state = self.ADJUST if np.linalg.norm(self.waypoint) > self.position_tol else self.TURN
        self.entry = np.zeros(2)
        self.turn_ready = False
        self.change_state(state, f"gap beams={plan['beams']} opening={plan['opening']:.3f}m "
                          f"slack={plan['slack']:.3f}m goal={math.degrees(self.cross_heading):+.1f}deg "
                          f"adjust={self.waypoint.round(3).tolist()} rear_clear_estimate={plan['travel']:.3f}m "
                          f"edges={self.edges.round(3).tolist()}")

    def stalled(self, key, remaining, improvement):
        if self.progress_key != key or remaining < self.progress_value - improvement:
            self.progress_key, self.progress_value, self.progress_time = key, remaining, self.now
        return self.now - self.progress_time > self.progress_timeout

    def waited(self, key):
        """True once a state has been holding still on `key` past the timeout."""
        return self.stalled(key, 0.0, 0.0)

    def abort_gap(self, reason):
        if self.edges is not None:
            self.failed.append((self.now + self.failed_timeout, self.edges.copy()))
            # Sweep back across the obstacle rather than continuing away from it:
            # the rejected gap is suppressed for failed_gap_timeout, so the leg
            # resumes past it instead of re-latching the same opening.
            self.side *= -1
            self.last_side_switch = self.now
        # Inside a narrow opening, return along the crossing to the known turning
        # position before restoring heading. Search must never rotate inside a gap.
        next_state = self.RECOVER if self.bubble_clear() else self.RETREAT
        self.change_state(next_state, reason)
        return 0.0, 0.0, 0.0

    def gap_hint(self):
        """Side of a visible opening too far off to reach yet, or 0 if none."""
        best, hint = math.inf, 0
        for a, b in self.edge_pairs()[0]:
            measured = self.pair_edges(a, b)[1]
            if float(np.linalg.norm(measured[1] - measured[0])) < 2 * self.half_width + self.min_slack:
                continue
            midpoint = measured.mean(axis=0)
            world = wrap(math.atan2(midpoint[0], midpoint[1]) + self.yaw)
            if math.cos(world) <= 0.0:
                continue
            offset = float(np.linalg.norm(midpoint))
            if offset < best and abs(math.sin(world)) > 1e-3:
                best, hint = offset, (1 if math.sin(world) > 0.0 else -1)
        return hint

    def search(self):
        front = self.bubble_free(-self.yaw)
        if self.state == self.FORWARD:
            if front > self.stop + self.position_tol:
                return -self.yaw, self.max_v, 0.0
            left = self.bubble_free(-math.pi / 2 - self.yaw)
            right = self.bubble_free(math.pi / 2 - self.yaw)
            # Strafe toward an opening that is visible but still out of reach;
            # clearance only decides the direction when no gap is in sight.
            hint = self.gap_hint()
            self.side = hint if hint else (1 if right >= left else -1)
            self.last_side_switch = self.now
            self.change_state(self.SEARCH, f"front={front:.3f}m side={'right' if self.side > 0 else 'left'} "
                              f"left={left:.3f}m right={right:.3f}m "
                              f"{'toward visible gap' if hint else 'no gap in sight'}")
            return 0.0, 0.0, 0.0
        if front >= self.resume:
            self.change_state(self.FORWARD, f"forward bubble path reopened: {front:.3f}m")
            return 0.0, 0.0, 0.0
        if self.bubble_clear():
            gaps = self.find_gaps()
            if gaps:
                # Clearance wins; heading and adjustment length only break ties.
                best = max(gaps, key=lambda g: (g["slack"], -abs(wrap(g["beta"] + self.yaw)),
                                                -float(np.linalg.norm(g["origin"]))))
                self.latch_gap(best)
                return 0.0, 0.0, 0.0
        beta = wrap(self.side * math.pi / 2 - self.yaw)
        room = self.bubble_free(beta)
        margin = self.stop + self.position_tol
        if self.backing or room <= margin:
            other = self.bubble_free(wrap(-self.side * math.pi / 2 - self.yaw))
            # Overshooting the stop, or sitting in a gap mouth, leaves no room to
            # strafe either way; reversing side forever would never free it.
            if self.backing:
                self.backing = (front < self.stop + 2 * self.position_tol
                                or max(room, other) <= self.stop + 2 * self.position_tol)
            elif front < self.stop or (room <= margin and other <= margin):
                self.backing = True
                self.get_logger().info(f"SEARCH boxed in (front={front:.3f}m left/right="
                                       f"{min(room, other):.3f}m); backing off")
            if self.backing:
                rear_beta = wrap(math.pi - self.yaw)
                if self.bubble_free(rear_beta) > margin:
                    return rear_beta, self.side_v, 0.0
                self.backing = False
                self.debug("boxed", "SEARCH boxed in with no room behind either")
        if room <= margin:
            if self.now - self.last_side_switch >= 0.5:
                self.side *= -1
                self.last_side_switch = self.now
                self.get_logger().info(f"SEARCH reversing to {'right' if self.side > 0 else 'left'}; "
                                       f"side clearance={room:.3f}m")
            return 0.0, 0.0, 0.0
        return beta, self.side_v, 0.0

    def handle_state(self):
        if self.state in (self.FORWARD, self.SEARCH):
            return self.search()
        if self.state == self.ADJUST:
            remaining = float(np.linalg.norm(self.waypoint))
            if remaining <= self.position_tol:
                if np.linalg.norm(self.velocity) > 1e-4:
                    return 0.0, 0.0, 0.0
                self.change_state(self.TURN, "adjustment complete")
                return 0.0, 0.0, 0.0
            beta = math.atan2(self.waypoint[0], self.waypoint[1])
            if self.bubble_free(beta) <= remaining or self.stalled(self.ADJUST, remaining, self.progress_dist):
                return self.abort_gap("adjustment blocked or stalled")
            return beta, min(self.side_v, remaining / 0.3), 0.0
        if self.state in (self.TURN, self.RECOVER):
            if self.state == self.TURN and not self.turn_ready:
                # Braking a strafe changes the approach position. Freeze the turn
                # bearing only once stopped, using the gap in the current frame.
                if np.linalg.norm(self.velocity) > 1e-4 or abs(self.omega) > 1e-4:
                    return 0.0, 0.0, 0.0
                # The edges were latched from afar with coarse angular sampling;
                # re-measure them from here before committing to a heading.
                self.refresh_edges()
                plan = self.gap_plan(self.edges, np.zeros(2))
                if plan is None:
                    return self.abort_gap(f"crossing rejected before turn: {self.plan_reason}")
                self.cross_heading = wrap(plan["beta"] + self.yaw)
                self.entry = np.zeros(2)
                self.turn_ready = True
            target = self.cross_heading if self.state == self.TURN else 0.0
            error = wrap(target - self.yaw)
            if not self.bubble_clear():
                if self.state == self.TURN:
                    return self.abort_gap("turning bubble no longer clear")
                if self.entry is not None:
                    self.change_state(self.RETREAT, "recovery bubble blocked; returning to entry")
                elif self.waited("recover-blocked"):
                    # Never wait here indefinitely: searching keeps the heading and
                    # can strafe or back away, which is what frees the bubble.
                    self.edges = self.waypoint = self.entry = None
                    self.change_state(self.SEARCH, "recovery bubble stayed blocked; searching instead")
                else:
                    self.debug("recover-blocked", "RECOVER waiting for turning-bubble clearance")
                return 0.0, 0.0, 0.0
            if abs(error) <= self.align_tol and abs(self.omega) < 0.01:
                if self.state == self.RECOVER:
                    self.edges = self.waypoint = self.entry = None
                    self.change_state(self.SEARCH, "original heading restored")
                else:
                    if not self.crossing_clear(error):
                        return self.abort_gap("crossing entry blocked after turn")
                    self.change_state(self.CROSS, "aligned; switching from circle to rectangular footprint")
                return 0.0, 0.0, 0.0
            if self.stalled(self.state, abs(error), math.radians(0.1)):
                if self.state == self.TURN:
                    return self.abort_gap("turn stalled")
                self.progress_key = None  # keep trying to recover, never start another turn target
            turn = math.copysign(min(self.max_w, self.yaw_gain * abs(error),
                                     math.sqrt(2 * self.alpha * abs(error))), error)
            return 0.0, 0.0, turn
        if self.state == self.CROSS:
            beta = wrap(self.cross_heading - self.yaw)
            u = direction(beta)
            reach = self.half_length * abs(math.cos(beta)) + self.half_width * abs(math.sin(beta))
            rear_clear = float(np.max(self.edges @ u)) + reach
            if rear_clear < 0.0:
                self.edges = self.waypoint = self.entry = None
                self.change_state(self.EXIT, "rear cleared; heading recovery handled after crossing")
                return 0.0, 0.0, 0.0
            if self.rectangle_free(beta) <= self.stop + 0.005:
                return self.abort_gap("crossing blocked")
            if self.stalled(self.CROSS, rear_clear, self.progress_dist):
                return self.abort_gap("crossing stalled")
            return beta, self.side_v, 0.0
        if self.state == self.EXIT:
            if self.bubble_clear():
                self.change_state(self.RECOVER, "turning bubble available beyond gap")
                return 0.0, 0.0, 0.0
            # Keep the nose fixed beyond the gap. A missing turning bubble is not
            # a failed crossing and must not send the robot back through the gap.
            beta = wrap(self.cross_heading - self.yaw)
            room = self.rectangle_free(beta)
            if room <= self.stop + 0.005:
                # Boxed in beyond the gap with no room to turn: reverse along the
                # crossing heading, the one direction known to have been clear.
                back = wrap(beta + math.pi)
                if self.waited("exit-hold") and self.rectangle_free(back) > self.stop + 0.005:
                    self.debug("exit", f"EXIT blocked ahead ({room:.3f}m) with no turning bubble; "
                               "reversing along the crossing heading")
                    return back, self.side_v, 0.0
                self.debug("exit", f"EXIT gap crossed; holding heading, forward clearance={room:.3f}m "
                           "and turning bubble not yet clear")
                return 0.0, 0.0, 0.0
            self.debug("exit", f"EXIT continuing with fixed heading; forward clearance={room:.3f}m")
            return beta, self.side_v, 0.0
        if self.state == self.RETREAT:
            if np.linalg.norm(self.velocity) < 1e-4 and self.bubble_clear():
                self.change_state(self.RECOVER, "turning bubble regained on retreat")
                return 0.0, 0.0, 0.0
            if self.entry is None or np.linalg.norm(self.entry) <= self.position_tol:
                if self.waited("retreat-blocked"):
                    # The entry pose no longer offers a turning bubble; stop
                    # waiting for one and let the search free the robot.
                    self.edges = self.waypoint = self.entry = None
                    self.change_state(self.SEARCH, "retreat reached entry without a turning bubble")
                    return 0.0, 0.0, 0.0
                self.debug("retreat-blocked", "RETREAT waiting: no clear turning position reached")
                return 0.0, 0.0, 0.0
            beta = math.atan2(self.entry[0], self.entry[1])
            return beta, min(self.side_v, float(np.linalg.norm(self.entry)) / 0.3), 0.0
        return 0.0, 0.0, 0.0

    def crossing_clear(self, beta):
        """After turning, require room to start; leave later checks to CROSS."""
        free = self.rectangle_free(beta)
        minimum = self.stop + 0.005
        self.debug("cross-check", f"CROSS entry free={free:.3f}m minimum={minimum:.3f}m "
                   f"heading_error={math.degrees(beta):+.2f}deg; clearance checked during travel")
        return free > minimum

    # Pure translation and pure rotation; integrate only the command actually sent.
    def track(self, points, dt):
        p = points - self.velocity * dt
        angle = self.omega * dt
        cs, sn = math.cos(angle), math.sin(angle)
        return np.column_stack((p[:, 0] * cs - p[:, 1] * sn,
                                p[:, 1] * cs + p[:, 0] * sn))

    def publish_motion(self, beta, speed, turn):
        target = direction(beta) * max(0.0, speed)
        current_speed = float(np.linalg.norm(self.velocity))
        # Settle translation before turning, and settle before changing direction.
        if abs(turn) > 1e-9 or abs(self.omega) > 1e-9:
            target[:] = 0.0
        if current_speed > 1e-4 and speed > 0.0:
            if float(self.velocity @ direction(beta)) / current_speed < math.cos(math.radians(10)):
                target[:] = 0.0
        if current_speed > 1e-4:
            turn = 0.0
        delta = target - self.velocity
        step = self.accel * self.DT
        norm = float(np.linalg.norm(delta))
        velocity = self.velocity + delta * min(1.0, step / norm) if norm > 0.0 else self.velocity.copy()
        if np.linalg.norm(velocity) < 1e-5:
            velocity[:] = 0.0
        omega = self.omega + float(np.clip(turn - self.omega, -self.alpha * self.DT, self.alpha * self.DT))
        if np.linalg.norm(velocity) > 0.0:
            omega = 0.0
            actual_beta = math.atan2(velocity[0], velocity[1])
            rectangular = self.state in (self.CROSS, self.EXIT, self.RETREAT)
            room = self.rectangle_free(actual_beta) if rectangular else self.bubble_free(actual_beta)
            # v*DT + v^2/(2*a) must fit before the clearance boundary.
            usable = max(0.0, room - self.stop)
            limit = math.sqrt((self.accel * self.DT) ** 2 + 2 * self.accel * usable) - self.accel * self.DT
            sent_speed = float(np.linalg.norm(velocity))
            if sent_speed > limit:
                velocity *= limit / sent_speed
        if abs(omega) > 0.0 and not self.bubble_clear():
            omega = 0.0
            self.debug("turn-veto", "TURN held: an obstacle is inside the turning bubble")
        self.velocity, self.omega = velocity, omega
        msg = Twist()
        msg.linear.x, msg.linear.y = float(velocity[1]), float(-velocity[0])
        msg.angular.z = float(-omega)
        self.pub_cmd_vel.publish(msg)

    def timer_callback(self):
        if self.points is None:
            return
        self.now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.last_tick is None else min(max(self.now - self.last_tick, 0.0), 0.25)
        self.last_tick = self.now
        self.yaw = wrap(self.yaw + self.omega * dt)
        for name in ("edges", "waypoint", "entry"):
            value = getattr(self, name)
            if value is not None:
                moved = self.track(np.atleast_2d(value), dt)
                setattr(self, name, moved if value.ndim == 2 else moved[0])
        self.failed = [(expiry, self.track(pair, dt)) for expiry, pair in self.failed if expiry > self.now]
        # Keep the scan geometry in the same current frame as the tracked gap.
        cloud_dt = min(max(self.now - self.cloud_time, 0.0), 0.25)
        self.points = self.track(self.points, cloud_dt)
        self.beam_points = self.track(self.beam_points, cloud_dt)
        # The beam rays radiate from where the lidar was when the scan was taken.
        self.lidar_origin = self.track(np.atleast_2d(self.lidar_origin), cloud_dt)[0]
        self.cloud_time = self.now
        beta, speed, turn = self.handle_state()
        self.publish_motion(beta, speed, turn)
        self.debug("status", f"STATUS state={self.state} side={'right' if self.side > 0 else 'left'} "
                   f"yaw={math.degrees(self.yaw):+.1f}deg bubble_clear={self.bubble_clear()} "
                   f"front_circle_free={self.bubble_free(-self.yaw):.3f}m "
                   f"sent(right,forward,cw)=({self.velocity[0]:.3f},{self.velocity[1]:.3f},{self.omega:.3f}) "
                   f"scan_age={self.now - self.scan_time:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub_cmd_vel.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
