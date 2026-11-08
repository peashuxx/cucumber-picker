import argparse
import time

import servo_controller as sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--servo-id', type=int, default=1)
    parser.add_argument('--center', type=int, default=500)
    parser.add_argument('--delta', type=int, default=80)
    parser.add_argument('--run-time-ms', type=int, default=600)
    args = parser.parse_args()

    low = max(0, args.center - args.delta)
    high = min(1000, args.center + args.delta)

    print(f'move servo {args.servo_id}: {args.center} -> {low} -> {high} -> {args.center}')
    sc.setServoPulse(args.servo_id, args.center, args.run_time_ms)
    time.sleep(args.run_time_ms / 1000.0 + 0.2)
    sc.setServoPulse(args.servo_id, low, args.run_time_ms)
    time.sleep(args.run_time_ms / 1000.0 + 0.2)
    sc.setServoPulse(args.servo_id, high, args.run_time_ms)
    time.sleep(args.run_time_ms / 1000.0 + 0.2)
    sc.setServoPulse(args.servo_id, args.center, args.run_time_ms)
    time.sleep(args.run_time_ms / 1000.0 + 0.2)
    print('done')


if __name__ == '__main__':
    main()
