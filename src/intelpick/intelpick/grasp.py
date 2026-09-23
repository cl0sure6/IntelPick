"""Waypoints for one pick. Pure geometry, no ROS.

The RoArm-M2-S is 4-DOF (base, shoulder, elbow, clamp). XYZ uses up the first three joints,
so the clamp's pitch is whatever the inverse kinematics gives: it cannot be ordered to point
straight down. Two strategies:

  radial   (default for M2-S): drop to grasp height a little closer to the base, then slide
           outward along the base->object line so the object enters the open jaws from the side.
  vertical (M3, or if tests show the M2-S clamp reaches down cleanly): hover above, descend.

Each waypoint is (x, y, z, slow). `slow` marks the moves near the object.
"""

import math


def pick_waypoints(x, y, z_grasp, z_hover, approach='radial', standoff=0.04, offset=0.0):
    """
    standoff: radial only, how far short of the object (towards the base) to descend, metres.
    offset:   how far past the object centroid (away from the base) the clamp tip should end,
              so the object sits inside the jaws instead of at the tips. Tune on hardware.
    """
    r = math.hypot(x, y)
    if r < 1e-6:
        raise ValueError('object at the arm base')
    ux, uy = x / r, y / r
    gx, gy = x + offset * ux, y + offset * uy  # clamp tip target
    if approach == 'vertical':
        return [
            (gx, gy, z_hover, False),
            (gx, gy, z_grasp, True),
        ]
    if approach == 'radial':
        sx, sy = x - standoff * ux, y - standoff * uy
        return [
            (sx, sy, z_hover, False),
            (sx, sy, z_grasp, True),
            (gx, gy, z_grasp, True),
        ]
    raise ValueError(f'unknown approach {approach!r}')


def min_pick_radius(min_reach, approach, standoff):
    """Closest object the approach can handle without dipping inside min_reach."""
    return min_reach + (standoff if approach == 'radial' else 0.0)
