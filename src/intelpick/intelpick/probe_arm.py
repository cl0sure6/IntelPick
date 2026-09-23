"""First contact with a real RoArm: answers the questions the docs don't (milestone M1).

    ros2 run intelpick probe_arm --port /dev/ttyUSB0

Every motion waits for ENTER, so keep a hand near the power switch and the table clear.
Steps: feedback -> home -> does T:104 block? -> gripper angles (empty vs holding)
       -> table height (move the limp arm by hand) -> reach at table height.
Prints the config values it measured at the end.
"""

import argparse
import math
import time

from .roarm import RoArm


def ask(msg):
    input(f'\n>>> {msg} [ENTER] ')


def fmt(fb):
    joints = ' '.join(f'{k}={fb[k]:.2f}' for k in ('b', 's', 'e', 't', 'r', 'g') if k in fb)
    return f'xyz=({fb["x"]:.1f}, {fb["y"]:.1f}, {fb["z"]:.1f}) mm  {joints}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--host', default='', help='arm IP, to use Wi-Fi/HTTP instead of serial')
    ap.add_argument('--model', default='m2', choices=['m2', 'm3'])
    ap.add_argument('--clearance', type=float, default=20.0,
                    help='height above the table for the reach test, mm')
    ap.add_argument('--grip-torque', type=int, default=30, help='%% of servo max, as in arm_node')
    ap.add_argument('--closed', type=float, default=3.1, help='gripper_closed from the config')
    ap.add_argument('--dry-run', action='store_true', help='simulated arm, to rehearse the steps')
    args, _ = ap.parse_known_args()

    arm = RoArm(port=args.port, host=args.host, model=args.model, dry_run=args.dry_run)
    arm.connect()
    found = {}
    try:
        # 1. Link
        fb = arm.get_feedback(timeout=2.0)
        if fb is None:
            raise SystemExit('no feedback. Check: usbipd attach, `ls /dev/ttyUSB*`, dialout group, '
                             'arm powered (12 V), nothing else holding the port (arm_node?)')
        print('link ok:', fmt(fb))
        if 'v' in fb:
            volts = fb['v'] / 100
            print(f'supply {volts:.2f} V' + ('  <-- LOW, expect weak/erratic moves' if volts < 11 else ''))

        # 2. Home
        ask('home the arm (T:100)')
        t0 = time.monotonic()
        print('homed' if arm.home() else 'home did not settle', f'in {time.monotonic() - t0:.1f}s')
        home = arm.get_feedback()
        print('home pose:', fmt(home))

        # 3. Does T:104 block the feedback channel while moving?
        ask('move 40 mm down from home, slowly')
        target = (home['x'], home['y'], home['z'] - 40)
        arm.send(arm.xyz_cmd(*target, spd=0.1))
        t0 = time.monotonic()
        replies, arrived = [], None
        while time.monotonic() - t0 < 8:
            f = arm.get_feedback(timeout=0.5)
            if f:
                replies.append(time.monotonic() - t0)
                if arrived is None and math.dist((f['x'], f['y'], f['z']), target) < 8:
                    arrived = time.monotonic() - t0
        during = [r for r in replies if arrived is None or r < arrived]
        print(f'arrived after {arrived:.1f}s' if arrived else 'did not arrive within 8 s')
        print(f'feedback replies while moving: {len(during)} '
              f'({"T:104 does NOT block feedback" if len(during) > 2 else "T:104 blocks feedback"})')

        # 4. Gripper, at the same torque limit arm_node will use
        arm.set_grip_torque(args.grip_torque)
        ask('open the gripper')
        found['open'] = arm.set_gripper(1.6)
        print(f'open: measured {found["open"]}')
        ask('close it EMPTY')
        found['empty'] = arm.set_gripper(args.closed)
        print(f'closed empty: measured {found["empty"]}')
        arm.set_gripper(1.6)
        ask('hold a cube/cap between the jaws, then ENTER to close on it')
        found['held'] = arm.set_gripper(args.closed)
        print(f'closed on object: measured {found["held"]}')
        arm.set_gripper(1.6)

        # 5. Table height, by hand
        ask('torque OFF: support the arm, then press ENTER')
        arm.set_torque(False)
        ask('move the clamp tip down onto the table by hand, hold it there')
        fb = arm.get_feedback()
        found['table_z'] = fb['z']
        print('at table:', fmt(fb))
        ask('keep holding it; torque will come back ON and the arm will hold this pose')
        arm.set_torque(True)

        # 6. Reach at table height, straight ahead
        z = found['table_z'] + args.clearance
        reached = []
        for r in (120, 160, 200, 250, 300, 350, 400):
            ask(f'move to x={r} mm, y=0, z={z:.0f} mm ({args.clearance:.0f} mm above table). '
                'Watch how the clamp meets the table')
            ok = arm.move_to(r, 0, z, spd=0.15, timeout=8)
            fb = arm.get_feedback()
            print(('reached ' if ok else 'NOT reached ') + (fmt(fb) if fb else ''))
            if ok:
                reached.append(r)
            else:
                break
        found['reach'] = reached
        ask('home')
        arm.home()
    except (KeyboardInterrupt, EOFError):
        print('\naborted')
    finally:
        arm.set_torque(True)
        arm.close()

    print('\n=== measured, copy into config/intelpick.yaml ===')
    if found.get('empty') and found.get('held'):
        # The sorter calls a grip empty when angle >= gripper_closed - margin: put that threshold
        # halfway between the two measured angles.
        threshold = (found['empty'] + found['held']) / 2
        print(f'empty_grip_margin: {args.closed - threshold:.2f}   '
              f'# empty {found["empty"]:.2f} vs held {found["held"]:.2f}')
        if found['empty'] - found['held'] < 0.1:
            print('  WARNING: empty and held angles are close; grasp check will be unreliable')
    if 'table_z' in found:
        print(f'table_z: {found["table_z"] / 1000:.3f}   # calibrate will refine this')
    if found.get('reach'):
        print(f'min_reach: {found["reach"][0] / 1000:.2f}\nmax_reach: {found["reach"][-1] / 1000:.2f}')
    print('Also note how the clamp met the table: if it could come down from above, '
          'try approach: vertical; otherwise keep radial.')
