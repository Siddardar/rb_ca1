# Gap-finding obstacle avoidance — plan

## Context

`src/rb2301_ca1/rb2301_ca1/obstacle_avoidance.py` builds three fixed corridors (front at
robot width, left/right at robot length) and strafes to a latched side. It cannot see
diagonal routes and cannot use the fact that the robot is narrower than it is long.

A previous attempt replaced the whole controller at once and could not find gaps. Travel
direction and nose direction are independent on a holonomic base, and getting the relationship
between them wrong is what broke it.

Everything numeric below is geometry, not tuning: the constants are parameters and the
thresholds are derived from the footprint rather than chosen. Where a worked number appears it
uses `W = 0.225`, `L = 0.289` purely as illustration.

---

## Frame and sign conventions

Bearings and yaw are both measured **clockwise from the nose**.

```
                    +y   nose, beta = 0
                     |
                     |
       beta = -90 ---R--- +x   beta = +90  (robot's right)
                     |
                    -y   beta = 180

u(beta) = ( sin beta , cos beta )
```

`psi` = rotation accumulated since startup, clockwise-positive. A body bearing `beta` is world
bearing `beta + psi`. The goal is world bearing 0.

**The sign trap.** ROS is counterclockwise-positive and REP-103 fixes the Twist axes, so every
conversion flips:

```
twist.linear.x  =  v * cos(beta)        # along the nose
twist.linear.y  = -v * sin(beta)        # to the left  (beta is clockwise)
twist.angular.z = -dpsi/dt              # angular.z is CCW, psi is CW

psi += -angular_z_published * dt        # dead reckoning
unwind: angular.z = +yaw_gain * psi     # to drive psi toward 0
```

Worked check: turn right 30 deg, so `psi = +30`. The startup direction is now 30 deg to the
robot's *left*, i.e. `beta = -psi = -30`. Unwinding means turning left, so `angular.z > 0`, and
`+yaw_gain * psi` is positive. Consistent.

**Body-relative vs startup-relative.** Anything computed from the point cloud is already
body-relative. Never subtract `psi` from it. If `beta_t` is a body bearing you want to face:

```
alpha      = beta_t                 rotation still needed
psi_target = wrap(psi + beta_t)     the same target, startup-relative
```

`wrap(beta_t - psi)` is wrong: with `psi = +30` and a target dead ahead the required turn is 0,
but that expression gives -30.

---

## Part 1 — Gap finding

### Step 1: bearings

The sensor reports counterclockwise angles in its own frame, so:

```
beta_i = -( msg.angle_min + i * msg.angle_increment + lidar_mount_yaw )
```

`lidar_mount_yaw` is a parameter. Set it by putting one object dead ahead and reading the
bearing of the scan minimum; it should come out near 0.

Cache `cos beta_i` and `sin beta_i`. Across scans only `r_i` changes, so the trig is computed
once. Invalidate on any change to `(len(ranges), angle_min, angle_increment)` — not length
alone, since a sensor can report the same count at a different increment.

Clean invalid returns as the current file does (`obstacle_avoidance.py:67-68`): non-finite or
below `range_min` becomes `range_max`, which reads as free space.

### Step 2: find discontinuities

Adjacent-beam chords are useless alone. A beam that shoots *through* a gap returns a long
range, and the chord to its endpoint is huge without being an opening.

Walk the circular scan. With `J_i = r_{i+1} - r_i`:

```
mark an edge where |J_i| > W/2      # smaller jumps cannot hide a passage
  J_i > 0  (range rises)  ->  near point is beam i
  J_i < 0  (range falls)  ->  near point is beam i+1
```

Only the **near side** of a discontinuity is a gap edge.

### Step 3: pair edges

For each rising edge at index `p`, scan forward to the next falling edge at index `q`:

```
P_a = beam p        P_b = beam q+1
```

The beams strictly between them see through the gap. Two guards: cap the angular span so you
never pair across half the scan, and require the between-beams to be farther than both edges.

### Step 4: quick reject

```
g = |P_b - P_a| = sqrt( r_a^2 + r_b^2 - 2*r_a*r_b*cos(dbeta) )
```

`g` is the true opening and bounds everything: the projected opening in Step 5 is never larger.
So `g <= W + clearance` rejects the pair outright, at any orientation and any approach.

`cos` is even so the sign of `dbeta` does not matter here, but wrap consistently anyway.

Do **not** introduce an "effective width" `2*sqrt(r_a*r_b)*sin(dbeta/2)`. It is yaw-invariant,
so it cannot describe what the robot fits through, and it is a lossy proxy for a quantity Step 5
computes exactly.

### Step 5: the crossing test

A straight crossing is fully described by two things: the travel direction and the nose
direction. Take the travel direction as the bearing to the chord midpoint:

```
M      = (P_a + P_b) / 2
beta_M = bearing of M
u_hat  = ( sin beta_M , cos beta_M )
n_hat_u = ( -cos beta_M , sin beta_M )        perpendicular to travel
```

The opening only counts in the direction across the path, so project the chord:

```
opening_perp = | (P_b - P_a) . n_hat_u |
```

and compare it against the corridor the body sweeps, where `delta` is the travel-vs-nose angle:

```
w(delta) = W*|cos delta| + L*|sin delta|

fit:  w(delta) + clearance < opening_perp
```

That is the whole test. `w()` is the same function the general corridor check uses, so nothing
new is introduced. Degenerate case checks out: travelling along the chord gives
`opening_perp = 0`, which never fits — you cannot pass through a slot by moving along it.

**Point the nose along the travel direction.** `w(delta)` is minimised at `delta = 0`, giving
`W`. Writing `mu` for the angle between travel and the gap's normal, the exact requirement is

```
required(nu, mu) = [ L*|sin(nu-mu)| + W*|cos(nu-mu)| ] / |cos mu|
```

for a nose at `nu` from the normal, and its minimum over `nu` is at `nu = mu` — nose on travel —
for every `mu`, giving `W / cos(mu)`. Facing the gap *normal* instead is much worse:

```
mu        nose on travel      nose on normal
10 deg        0.2285             0.2760
20 deg        0.2394             0.3302
30 deg        0.2598             0.3919
45 deg        0.3182             0.5140
```

Because `sec` is flat near zero, obliquity is cheap. A 0.30 m opening tolerates **41.4 deg** of
mismatch, 0.35 m tolerates 50 deg. Any rule that rejects gaps at 15-25 deg of obliquity is
throwing away most usable gaps.

Do not construct the gap normal at all. It is not needed for the orientation target (which is
the travel direction) and not needed for the fit test (which uses `opening_perp`).

### Step 6: the decision

Since the body frame is nose-aligned, not rotating means `delta = beta_M`:

```
g <= W + clearance                      ->  reject, impossible at any orientation
w(beta_M) + clearance < opening_perp    ->  crossable as-is.  needs_rotation = False
W + clearance < opening_perp            ->  crossable if faced. needs_rotation = True
otherwise                               ->  reject
```

The second branch is preferred when available: it costs no rotation and no heading. Record
`needs_rotation` on the latch — an approach accepted *because* it needs no rotation must not
then be made to rotate, or it comes to depend on turning space it never required.

Score surviving candidates by deviation from the goal, `|wrap(beta_M + psi)|`, and take the
smallest. Clearance from Part 3 enters as a constraint, never as a reward.

---

## Part 2 — Execution

Selection and execution have to be separated, or approaching, crossing and heading recovery all
fire at once. Default state is CRUISE; a crossing is a latched excursion from it.

```
CRUISE   no latch. Choose a travel direction from the swept-distance grid (Part 3.1):
         feasible, minimising |wrap(beta + psi)|, with the room and hysteresis terms
         from Part 4. Heading recovery active. This is the ordinary case -- open space
         and lone obstacles produce no gap pair at all, and must still be driven.

SELECT   runs inside CRUISE each tick. If Steps 2-6 yield a feasible candidate, latch it
         and leave CRUISE. Otherwise stay. No candidate is not a reason to stop moving.

ALIGN    only if needs_rotation. Rotate the nose toward Theta*. Translate no further
         than the exact sweep permits. No re-selection, no heading recovery.

CROSS    drive the latched direction. No re-selection, no heading recovery.

CLEAR    both tracked edges lie behind the trailing edge -> release, return to CRUISE.
```

### What the latch stores

```
Theta*        = wrap(beta_M + psi)      the crossing direction, STARTUP-relative
edge points   P_a, P_b                  tracked, in body coordinates
needs_rotation                          from Step 6
```

**The crossing direction must be frozen, not recomputed.** Storing it startup-relative and
recovering the body-frame command as

```
delta = wrap(Theta* - psi)
```

is what makes it survive. Recomputing it as the bearing to the current midpoint does not: as the
robot nears `M` that bearing is ill-conditioned, because `|M| -> 0`, and once past it the bearing
points roughly backwards. A controller chasing it turns around with its tail still in the
opening.

If `needs_rotation`, ALIGN drives `psi` until `delta ~ 0` and CROSS then runs nose-first. If not,
ALIGN is skipped, `psi` is untouched, and CROSS runs at the constant nonzero `delta` the strafe
was accepted with. `w(delta)`, the sweep check and `h(delta)` all already take `delta`, so the
same formulas serve both.

### Tracking the edges, and why detection must not gate release

Track `P_a` and `P_b` by applying the commanded increment each tick — the same transform as the
sampled check in Part 3.3, one step at a time. With the published twist:

```
d_body = ( -linear.y * dt , linear.x * dt )      in (right, nose)
dpsi   = -angular.z * dt                        clockwise-positive

p    <- p - d_body                              translate in the old axes
beta <- wrap(beta - dpsi)                       then re-express in the turned axes
```

Optionally snap each tracked point to the nearest actual return within a small gate, keeping the
prediction when nothing falls inside it. A crossing lasts under a second or so, so this
integration is bounded by construction and drift never accumulates across crossings.

**Losing the detector candidate must not release the latch.** As the robot enters the opening its
two edges spread toward opposite sides of the scan and stop satisfying the angular-span and
pairing rules of Steps 2-3. That is what a *successful* crossing looks like to the detector, so
dropping the latch on association failure would fire at exactly the wrong moment. Track the
bordering obstacles; do not ask whether they still form a newly detected gap.

### When the rear has cleared

Computable from the tracked points alone. For each, with `f = p . u_hat`, the footprint's
trailing edge along travel sits at `-h(delta)` where

```
h(delta) = ( L*|cos delta| + W*|sin delta| ) / 2
```

The edge is behind the robot when `f < -h(delta)`. Require it for **both** edges. This is the
condition that stops heading recovery starting while the tail is still between the obstacles.
Back it with a timeout so a latch can never persist indefinitely.

### Abort

A different predicate from selection: the exact swept distance falling below the clearance
margin, not "a better gap appeared". One predicate serving both is what makes a reactive
controller thrash.

`opening_perp` is deliberately **not** re-derived during the crossing. The swept-distance check
already constrains the motion continuously and against the tracked points, so re-projecting a
frozen direction onto refreshed geometry would add nothing but a second opinion that can
disagree with itself.

Heading recovery is suppressed during ALIGN and CROSS and resumes on release. Clearance is
checked every tick in every state.

## Part 3 — Collision geometry

Three distinct quantities. Conflating any two is a bug, so keep them as separate named
functions with comments saying why.

### 1. Exact swept distance for a straight translation

In the body frame the footprint is the **axis-aligned** box `[-W/2, W/2] x [-L/2, L/2]` in
`(right, nose)`. Nothing rotates — the travel direction tilts, the box does not. So for an
obstacle point `p` translating along `u_hat`, the collision distance is a two-slab ray-box
intersection, solving `p - t*u_hat` inside the box:

```
for each axis i with half-extent e_i:
    if |u_i| ~ 0:  no constraint if |p_i| <= e_i, otherwise never enters
    else:          t in [ (p_i - e_i)/u_i , (p_i + e_i)/u_i ]   (ordered)

t_enter = max of the lower bounds      t_exit = min of the upper bounds
hit iff t_enter <= t_exit and t_exit >= 0,  at distance max(t_enter, 0)
```

Exact, vectorises over all beams, and returns a **distance** rather than a boolean, which is
what speed control needs. The minimum over beams is the free travel along `u_hat`.

CRUISE needs this over a grid of candidate directions, which broadcasts to `(beams, directions)`
— two axes, two slab bounds each. At 720 beams and a few degrees of resolution that is a few
hundred thousand elements per tick, and no trigonometry beyond the cached beam table.

**Do not approximate this with a strip test.** `|l| < w(delta)/2 and f > -h(delta)` is the
bounding box of the rotated footprint, not the footprint, and the excess is worst exactly at
the diagonal travel this plan exists to enable:

```
travel   box/true area
   0 deg     1.00x
  15 deg     1.52x
  30 deg     1.89x
  45 deg     2.03x
```

Those rear corners make the robot refuse motion its body would clear. And the opposite
shortcut, `f > 0`, misses real collisions: translation leaves `l` invariant because `l` is the
perpendicular component, so a point inside the sweep stays inside, and being behind the centre
does not save you. Worked case at 45 deg travel: a point at `f = -0.0071`, `l = -0.1768` is
struck after 0.0106 m, yet `f > 0` discards it and the box strip flags it with no distance.

### 2. Distance to the footprint edge along a bearing

```
edge_dist(beta) = min( (L/2)/|cos beta| , (W/2)/|sin beta| )
```

Where a ray *exits* the rectangle. This is **not** a support function — support is the distance
to the supporting line and overstates the edge near the corners, 0.182 against 0.159 at 45 deg.
Substituting one for the other makes an overlap check report a violation with nothing
overlapping. Used for the rotation check below and as a standing overlap monitor.

### 3. Rotation, and rotation combined with translation

Checking that a translation is clear and that a rotation at the starting pose is clear does not
establish that doing both is clear — the corners follow different paths when they happen
together. Sample the motion and test the footprint at each sample, transforming the cloud into
each sampled pose:

```
p_k = R(-phi_k) * ( p - d_k )        then test against edge_dist
```

for displacement `d_k` and rotation `phi_k` at sample `k`. Eight samples over 720 beams is
negligible. Pure rotation is the special case `d_k = 0`, and it reduces to the swept union of
footprints over the angular interval — strictly larger than either endpoint, because the corners
trace arcs.

Do not substitute "room for a full spin". It is sound but far too conservative for small
corrections: unwinding a few degrees does not need space to turn all the way round, and
requiring it deadlocks the robot where it could comfortably straighten up.

---

## Part 4 — Heading recovery

```
psi += -angular_z_published * dt        integrate what was sent, after rate limiting
D(beta) = |wrap(beta + psi)|            deviation from the startup world heading
```

`D` is zero exactly at `beta = -psi`. Whatever orientation the robot is in, the cheapest heading
is the one continuing the original world direction, so "go straight if possible" and "return to
heading after a detour" are one expression rather than two rules. Translating along that heading
while commanding `angular.z = +yaw_gain * psi` holds a straight world line while the body
rotates upright underneath it.

Three things that must come with it:

- **Rate-limit `turn`.** The current `move_2D` clips `angular.z` but never slews it
  (`obstacle_avoidance.py:53`), so it can jump to full rate in one tick. `psi` integrates what
  was published, so an unslewed command is not integrable.
- **Clearance is a constraint, not a reward.** With `J = a*D - c*clearance`, reversing wins
  whenever `c*(c_back - c_fwd) > a*pi`, and `c_back` is near sensor range while `c_fwd` is near
  zero whenever something blocks you — so almost any `c` triggers it, and the cleaner the
  retreat the worse it gets. Saturate any room term at a horizon with `gamma * horizon < pi`,
  which makes the inequality unsatisfiable by construction.
- **Hysteresis in the world frame.** Penalise `|wrap(beta + psi - Theta_prev)|`, storing the
  previous choice in world terms. Penalising the body-frame change charges the robot for
  tracking a straight world line correctly while it rotates. Keep the weight below 1: with an L1
  penalty the minimiser sits exactly at one kink, so a weight above 1 freezes the old choice and
  the robot never returns to the goal.

Recovery is gated by the rotation check in Part 3, so it cannot swing a corner into something.

---

## Out of scope

- **Dead ends.** Assume a path forward exists. Escaping a pocket needs to reverse, and a
  memoryless policy on a non-convex free space has local minima regardless of tuning.
- **Passages needing a curved path** — where the route in and the route through differ enough
  that no single straight crossing works. That is the occupancy-grid / configuration-space case.
- **Repositioning before crossing.** Deliberately dropped. Its purpose was reducing the
  obliquity `mu`, but the penalty is `W/cos mu`, and a secant is flat near zero: 0.2285 at
  10 deg against 0.2250 at 0. It buys about a centimetre for metres of travel.

---

## Implementation order

1. Bearings and the frame/sign block.
2. Discontinuities and edge pairing (Steps 2-3).
3. Exact swept distance (Part 3.1) — the general clearance primitive, needed by everything else.
4. CRUISE plus heading recovery (Part 4). A complete, working controller on its own.
5. Crossing test (Steps 4-6).
6. `edge_dist` and the sampled rotation check (Parts 3.2, 3.3).
7. Latch, edge tracking and the phase machine (Part 2).

---

## Files

| path | change |
|---|---|
| `src/rb2301_ca1/rb2301_ca1/obstacle_avoidance.py` | all of it |

Footprint, clearance, gains, the discontinuity threshold, and `lidar_mount_yaw` should all be
ROS parameters — nothing should need a rebuild to retune.
