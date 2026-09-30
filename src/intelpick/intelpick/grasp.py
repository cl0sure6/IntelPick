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
from dataclasses import dataclass


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


@dataclass(frozen=True)
class Step:
    """One action of a pick or place. The sorter runs these through ROS services, the
    grasp_trial tool straight over serial, so both do exactly the same thing."""
    kind: str        # 'move' | 'open' | 'close'
    x: float = 0.0   # metres, arm frame (moves only)
    y: float = 0.0
    z: float = 0.0
    slow: bool = False
    what: str = ''   # for logs and the trial tool's prompts


_WAYPOINT_NAMES = {
    'vertical': ['above the object', 'down to grasp height'],
    'radial': ['above the stand-off point', 'down beside the object',
               'slide out onto the object'],
}


def pick_steps(x, y, z_grasp, z_hover, approach='radial', standoff=0.04, offset=0.0):
    """Open, approach, close, lift, close again.

    The second close re-reads the jaw angle after the lift: callers judge the grip by the
    angle from the LAST close. On the real M2-S a cube caught off-centre closed at 2.84 rad
    (looks held, threshold 3.04) and then slipped out while lifting; the jaws keep closing
    when that happens, so the post-lift reading catches it.
    """
    waypoints = pick_waypoints(x, y, z_grasp, z_hover, approach, standoff, offset)
    steps = [Step('open', what='open the gripper')]
    steps += [Step('move', wx, wy, wz, slow, name)
              for (wx, wy, wz, slow), name in zip(waypoints, _WAYPOINT_NAMES[approach])]
    gx, gy = waypoints[-1][:2]
    steps += [Step('close', what='close on the object'),
              # Lift before anything else, so a failed grasp never drags across the table.
              Step('move', gx, gy, z_hover, True, 'lift straight up'),
              Step('close', what='re-check the grip after the lift')]
    return steps


def place_steps(x, y, z_place, z_hover):
    return [Step('move', x, y, z_hover, False, 'over the drop point'),
            Step('move', x, y, z_place, True, 'down to release height'),
            Step('open', what='release'),
            Step('move', x, y, z_hover, False, 'back up')]


def is_empty_grip(angle, gripper_closed, margin):
    """Jaws closed almost fully = nothing between them. Unknown angle counts as not empty."""
    return angle is not None and not math.isnan(angle) and angle >= gripper_closed - margin
