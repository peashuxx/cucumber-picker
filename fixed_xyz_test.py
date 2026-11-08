import argparse
import sys
import time

import numpy as np

from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import get_ik
from kinematics import transform

import servo_controller as sc


def parse_pulse(text):
    values = [int(x) for x in text.split(',') if x.strip()]
    if len(values) != 5:
        raise ValueError('pulse must contain 5 values, for example 500,850,200,120,500')
    return values


def parse_groups(text):
    groups = []
    for group in text.split(';'):
        ids = [int(x) for x in group.split(',') if x.strip()]
        if ids:
            groups.append(ids)
    if not groups:
        raise ValueError('joint-groups must not be empty')
    return groups


def clamp_pulse(pulse):
    return [max(0, min(1000, int(x))) for x in pulse]


def fk_xyz(fk, pulse):
    angles = transform.pulse2angle(pulse)
    fk_res = fk.get_fk(angles)
    return fk_res[0]


def flatten_numbers(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        result = []
        for item in value:
            result.extend(flatten_numbers(item))
        return result
    return [float(value)]


def choose_ik_pulse(xyz, current_pulse, pitch):
    all_solutions = get_ik([float(i) for i in xyz], float(pitch), [-180.0, 180.0], 1.0)
    candidates = []
    for solution in all_solutions or []:
        for pulse in transform.angle2pulse(solution[0], True):
            pulse = clamp_pulse(pulse)
            delta = sum(abs(pulse[i] - current_pulse[i]) for i in range(5))
            candidates.append((delta, pulse, solution[0]))
    candidates.sort(key=lambda item: item[0])
    return candidates


def move_grouped(pulse, joint_groups, use_time_ms):
    for group in joint_groups:
        active = []
        for sid in group:
            if 1 <= sid <= 5:
                value = int(pulse[sid - 1])
                active.append((sid, value))
        if active:
            sc.setServoPulses(active, use_time_ms)
            print(f'move group packet {active}, time={use_time_ms}ms', flush=True)
            time.sleep(use_time_ms / 1000.0 + 0.15)


def move_slow(current_pulse, target_pulse, joint_groups, max_pulse_step, min_time_ms, time_per_100):
    current = list(current_pulse)
    target = clamp_pulse(target_pulse)
    max_delta = max(abs(target[i] - current[i]) for i in range(5))
    steps = max(1, int(np.ceil(max_delta / max_pulse_step)))
    fk = ForwardKinematics()

    for step in range(1, steps + 1):
        ratio = step / steps
        waypoint = [
            int(round(current[i] + (target[i] - current[i]) * ratio))
            for i in range(5)
        ]
        waypoint_delta = max(abs(waypoint[i] - current_pulse[i]) for i in range(5))
        use_time = int(max(min_time_ms, waypoint_delta * time_per_100 / 100.0))
        print(f'step {step}/{steps}: pulse={waypoint}, time={use_time}ms', flush=True)
        move_grouped(waypoint, joint_groups, use_time)
        current_pulse[:] = waypoint
        xyz = fk_xyz(fk, waypoint)
        print(f'  FK end -> X_base={xyz[0]:.3f} Y_base={xyz[1]:.3f} Z_base={xyz[2]:.3f}', flush=True)


def move_joint_sequential(current_pulse, target_pulse, joint_order, max_pulse_step, min_time_ms, time_per_100):
    fk = ForwardKinematics()
    target = clamp_pulse(target_pulse)
    current_pulse[:] = clamp_pulse(current_pulse)

    for sid in joint_order:
        if not 1 <= sid <= 5:
            continue
        idx = sid - 1
        start = int(current_pulse[idx])
        end = int(target[idx])
        delta = end - start
        steps = max(1, int(np.ceil(abs(delta) / max_pulse_step)))
        print(f'joint {sid}: {start} -> {end}, steps={steps}', flush=True)

        for step in range(1, steps + 1):
            ratio = step / steps
            value = int(round(start + delta * ratio))
            waypoint_delta = abs(value - current_pulse[idx])
            use_time = int(max(min_time_ms, waypoint_delta * time_per_100 / 100.0))
            current_pulse[idx] = value
            print(f'  joint {sid} step {step}/{steps}: pulse={current_pulse}, time={use_time}ms', flush=True)
            sc.setServoPulse(sid, value, use_time)
            time.sleep(use_time / 1000.0 + 0.2)
            xyz = fk_xyz(fk, current_pulse)
            print(f'    FK end -> X_base={xyz[0]:.3f} Y_base={xyz[1]:.3f} Z_base={xyz[2]:.3f}', flush=True)


def solve_and_maybe_move(name, xyz, current_pulse, args, joint_groups, fk):
    print(f'\n=== {name} ===', flush=True)
    print(f'target xyz  : X_base={xyz[0]:.3f} Y_base={xyz[1]:.3f} Z_base={xyz[2]:.3f}', flush=True)
    candidates = choose_ik_pulse(xyz, current_pulse, args.pitch)
    if not candidates:
        r = float(np.linalg.norm(xyz[:2]))
        d = float(np.linalg.norm(xyz))
        print(f'IK no solution: r_xy={r:.3f}, distance={d:.3f}, pitch={args.pitch:.1f}', flush=True)
        return current_pulse

    delta, target_pulse, angles = candidates[0]
    fk_target = fk_xyz(fk, target_pulse)
    print(f'candidate: delta={delta}, pulse={target_pulse}', flush=True)
    print(f'  FK check -> X_base={fk_target[0]:.3f} Y_base={fk_target[1]:.3f} Z_base={fk_target[2]:.3f}', flush=True)
    print(f'  angles   -> {[round(a, 2) for a in flatten_numbers(angles)]}', flush=True)

    if not args.move:
        return current_pulse

    if args.direct:
        print(f'direct move to target pulse={target_pulse}, time={args.direct_time_ms}ms', flush=True)
        move_grouped(target_pulse, joint_groups, args.direct_time_ms)
        current_pulse[:] = target_pulse
    elif args.joint_sequential:
        joint_order = [int(x) for x in args.joint_order.split(',') if x.strip()]
        move_joint_sequential(
            current_pulse,
            target_pulse,
            joint_order,
            args.max_pulse_step,
            args.min_move_time_ms,
            args.move_time_per_100_pulse,
        )
    else:
        move_slow(
            current_pulse,
            target_pulse,
            joint_groups,
            args.max_pulse_step,
            args.min_move_time_ms,
            args.move_time_per_100_pulse,
        )

    final_xyz = fk_xyz(fk, current_pulse)
    print(f'final FK -> X_base={final_xyz[0]:.3f} Y_base={final_xyz[1]:.3f} Z_base={final_xyz[2]:.3f}', flush=True)
    return current_pulse


def main():
    parser = argparse.ArgumentParser(description='Test IK and optional motion for a fixed base-frame xyz.')
    parser.add_argument('--xyz', help='base xyz in meters, for example 0.20,-0.03,0.45')
    parser.add_argument('--scan-base-axes', action='store_true',
                        help='test center/+X/-X/+Y/-Y/+Z/-Z around --scan-center')
    parser.add_argument('--scan-center', default='0.20,0.00,0.40',
                        help='center xyz for --scan-base-axes')
    parser.add_argument('--scan-step', type=float, default=0.05,
                        help='meter step for --scan-base-axes')
    parser.add_argument('--scan-wait-s', type=float, default=1.0,
                        help='pause after each moved scan point')
    parser.add_argument('--current-pulse', default='500,850,200,120,500',
                        help='current/initial 5-joint pulse used for nearest IK solution')
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--move', action='store_true', help='actually send servo commands')
    parser.add_argument('--open-gripper', action='store_true', help='open gripper before moving')
    parser.add_argument('--direct', action='store_true',
                        help='send the final IK pulse once instead of slow segmented motion')
    parser.add_argument('--direct-time-ms', type=int, default=2500,
                        help='move time used by --direct')
    parser.add_argument('--joint-groups', default='1;5;2;3;4')
    parser.add_argument('--joint-sequential', action='store_true',
                        help='move one joint to its final pulse before starting the next joint')
    parser.add_argument('--joint-order', default='1,5,2,3,4',
                        help='joint order used by --joint-sequential')
    parser.add_argument('--max-pulse-step', type=int, default=25)
    parser.add_argument('--min-move-time-ms', type=int, default=1800)
    parser.add_argument('--move-time-per-100-pulse', type=int, default=1800)
    parser.add_argument('--list-all', action='store_true', help='print all IK pulse candidates')
    args = parser.parse_args()

    current_pulse = parse_pulse(args.current_pulse)
    joint_groups = parse_groups(args.joint_groups)
    fk = ForwardKinematics()

    current_xyz = fk_xyz(fk, current_pulse)
    print(f'current pulse: {current_pulse}')
    print(f'current FK  : X_base={current_xyz[0]:.3f} Y_base={current_xyz[1]:.3f} Z_base={current_xyz[2]:.3f}')

    if args.open_gripper and args.move:
        print('open gripper: servo 10 -> 500')
        sc.setServoPulse(10, 500, 500)
        time.sleep(0.6)

    if args.scan_base_axes:
        center = np.array([float(x) for x in args.scan_center.split(',') if x.strip()], dtype=float)
        if len(center) != 3:
            raise ValueError('--scan-center must contain 3 values, for example 0.20,0,0.40')
        step = float(args.scan_step)
        points = [
            ('CENTER', center),
            ('BASE +X', center + np.array([step, 0.0, 0.0])),
            ('BASE -X', center + np.array([-step, 0.0, 0.0])),
            ('BASE +Y', center + np.array([0.0, step, 0.0])),
            ('BASE -Y', center + np.array([0.0, -step, 0.0])),
            ('BASE +Z', center + np.array([0.0, 0.0, step])),
            ('BASE -Z', center + np.array([0.0, 0.0, -step])),
        ]
        for name, point in points:
            current_pulse = solve_and_maybe_move(name, point, current_pulse, args, joint_groups, fk)
            if args.move:
                time.sleep(max(0.0, args.scan_wait_s))
        print('done')
        return

    if args.xyz is None:
        raise ValueError('provide --xyz or use --scan-base-axes')

    xyz = np.array([float(x) for x in args.xyz.split(',') if x.strip()], dtype=float)
    if len(xyz) != 3:
        raise ValueError('--xyz must contain 3 values, for example 0.20,-0.03,0.45')
    print(f'target xyz  : X_base={xyz[0]:.3f} Y_base={xyz[1]:.3f} Z_base={xyz[2]:.3f}')

    candidates = choose_ik_pulse(xyz, current_pulse, args.pitch)
    if not candidates:
        r = float(np.linalg.norm(xyz[:2]))
        d = float(np.linalg.norm(xyz))
        print(f'IK no solution: r_xy={r:.3f}, distance={d:.3f}, pitch={args.pitch:.1f}')
        return

    print(f'IK solutions: {len(candidates)}')
    shown = candidates if args.list_all else candidates[:3]
    for idx, (delta, pulse, angles) in enumerate(shown, start=1):
        fk_target = fk_xyz(fk, pulse)
        print(f'candidate {idx}: delta={delta}, pulse={pulse}')
        print(f'  FK check -> X_base={fk_target[0]:.3f} Y_base={fk_target[1]:.3f} Z_base={fk_target[2]:.3f}')
        print(f'  angles   -> {[round(a, 2) for a in flatten_numbers(angles)]}')

    target_pulse = candidates[0][1]
    if not args.move:
        print('dry-run only. Add --move to send servo commands.')
        return

    if args.direct:
        print(f'direct move to target pulse={target_pulse}, time={args.direct_time_ms}ms')
        move_grouped(target_pulse, joint_groups, args.direct_time_ms)
        final_xyz = fk_xyz(fk, target_pulse)
        print(f'final FK -> X_base={final_xyz[0]:.3f} Y_base={final_xyz[1]:.3f} Z_base={final_xyz[2]:.3f}')
        print('done')
        return

    moving_pulse = list(current_pulse)
    if args.joint_sequential:
        joint_order = [int(x) for x in args.joint_order.split(',') if x.strip()]
        move_joint_sequential(
            moving_pulse,
            target_pulse,
            joint_order,
            args.max_pulse_step,
            args.min_move_time_ms,
            args.move_time_per_100_pulse,
        )
        print('done')
        return

    move_slow(
        moving_pulse,
        target_pulse,
        joint_groups,
        args.max_pulse_step,
        args.min_move_time_ms,
        args.move_time_per_100_pulse,
    )
    print('done')


if __name__ == '__main__':
    main()
