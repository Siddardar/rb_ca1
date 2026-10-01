"""Writes a fixed obstacle world for CA1. Nothing random.

x is forward and y is left of where the robot starts. The robot drives up the page:

                 can can can can can can can can          <- front wall
        can                                      can       <- the two ways out are
        can                                       can         the corner gaps
        can                                       can
        can              ROBOT                    can      <- side walls
        can                                       can
        can                                       can

The front wall stops short of both side walls, and both side walls stop short of
the front wall, so the only ways out are the two corner gaps. Each corner gap is a
diagonal, and the two corners are cut differently, so the two ways out sit at
different angles and neither can be taken without turning first.

Edit the numbers below to move things. WRITE_WORLD = False leaves the world file
alone if you want to hand edit the sdf instead.
"""
import xml.etree.ElementTree as ET
import math
import os

WRITE_WORLD = True

# --- the pocket -------------------------------------------------------------
WALL_X = 1.4          # front wall, this far ahead of the robot
POCKET_HALF = 1.0     # side walls this far left and right
BACK_X = -0.8         # side walls run from here forward
SPACING = 0.2         # can spacing along every wall, 0.1 m of clear air between them

# --- the two corner gaps ----------------------------------------------------
# The front wall ends, and how far forward each side wall reaches. The distance
# between those two cans is the gap, and the diagonal between them is its angle.
LEFT_WALL_END = 0.75    # front wall stops here going left
LEFT_SIDE_END = 1.06    # left side wall reaches this far forward
RIGHT_WALL_END = -0.82  # front wall stops here going right
RIGHT_SIDE_END = 1.02   # right side wall reaches this far forward

CAN_RADIUS = 0.05
ROBOT_WIDTH = 0.20      # bare body, no margin

workspace_directory = os.path.dirname(os.path.realpath(__file__))[:-22]
overwrite_file = workspace_directory + '/rb2301_gz/worlds/obstacle_world_ca1.sdf'
obstacle_model = f'file:///{workspace_directory}/rb2301_gz/meshes/coke/6'


def line(start, end, spacing=SPACING):
    """Cans evenly along a line, both ends included, never more than spacing apart."""
    span = math.dist(start, end)
    steps = max(1, math.ceil(span / spacing))
    return [(round(start[0] + (end[0] - start[0]) * k / steps, 3),
             round(start[1] + (end[1] - start[1]) * k / steps, 3))
            for k in range(steps + 1)]


def build_layout():
    """The can centres, as a list of (x, y)."""
    return (line((WALL_X, RIGHT_WALL_END), (WALL_X, LEFT_WALL_END))       # front wall
            + line((BACK_X, POCKET_HALF), (LEFT_SIDE_END, POCKET_HALF))   # left wall
            + line((BACK_X, -POCKET_HALF), (RIGHT_SIDE_END, -POCKET_HALF)))  # right wall


def corner(wall_end, side_end, side):
    """The gap at one corner: its two cans, clear width, and the heading through it."""
    a = (WALL_X, wall_end)
    b = (side_end, side * POCKET_HALF)
    gap = math.dist(a, b) - 2 * CAN_RADIUS
    chord = (b[0] - a[0], b[1] - a[1])
    heading = math.degrees(math.atan2(-chord[0] * side, chord[1] * side))
    return a, b, gap, heading


def add_coke_element(x, y, n):
    obstacle = ET.Element("include")
    uri = ET.Element("uri")
    uri.text = obstacle_model
    obstacle.append(uri)
    name = ET.Element("name")
    name.text = f'coke{n}'
    obstacle.append(name)
    pose = ET.Element("pose")
    pose.text = f'{x} {y} 0 0 0 0'
    obstacle.append(pose)
    return obstacle


def generate_sdf_file():
    if not WRITE_WORLD:
        return

    cans = build_layout()
    print(f"Writing fixed obstacle world: {len(cans)} cans")
    for name, wall_end, side_end, side in (("left", LEFT_WALL_END, LEFT_SIDE_END, 1),
                                           ("right", RIGHT_WALL_END, RIGHT_SIDE_END, -1)):
        a, b, gap, heading = corner(wall_end, side_end, side)
        head_on = gap * abs(math.cos(math.radians(heading)))
        print(f"  {name} corner gap {gap:.3f} m between {a} and {b}, "
              f"drive it on a {heading:+.0f} deg heading: {head_on:.3f} m seen head on "
              f"against a {ROBOT_WIDTH:.2f} m robot")

    tree = ET.parse(overwrite_file)
    root = tree.getroot()
    world = root[0]
    for element in reversed(world):  # Remove all coke obstacles
        if element.tag == 'include':
            world.remove(element)
    tree.write(overwrite_file)

    for n, (x, y) in enumerate(cans, start=1):
        world.append(add_coke_element(x, y, n))

    tree.write(overwrite_file)


if __name__ == '__main__':
    generate_sdf_file()
