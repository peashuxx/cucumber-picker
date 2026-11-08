import argparse
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import onnxruntime as ort
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 运动学扩展由机械臂配套 SDK 提供。
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import get_ik
from kinematics import transform

import servo_controller as sc

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

class CucumberPickerNode(Node):
    def __init__(self, init_pulse=None, init_run_time_ms=1500, skip_init=False,
                 enable_pick=False, pick_pitch=0.0, pick_cooldown_s=6.0,
                 color_topic='/camera/camera/color/image_raw',
                 depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
                 sync_slop=0.3, preview_port=8080, preview_quality=80,
                 conf_threshold=0.75, target_offset=None,
                 min_camera_depth=0.08, max_camera_depth=0.80,
                 depth_roi_size=9, detect_every_n=3, lock_frames=5,
                 lost_timeout=2.0, single_shot=False, approach_only=False,
                 coord_log_interval=1.0, max_pulse_step=40,
                 move_time_per_100_pulse=1000, min_move_time_ms=1200,
                 approach_height=0.05, joint_groups=None,
                 joint_sequential=False, joint_order=None,
                 cartesian_motion=False, cartesian_steps=6,
                 direct_target=False, gripper_open_pulse=500, gripper_close_pulse=590,
                 post_grip_servo1_delta=375, post_grip_time_ms=1500,
                 camera_ee_offset=None,
                 freeze_after_lock=True, reset_after_motion=True):
        super().__init__('cucumber_picker_node')
        self.bridge = CvBridge()
        self.fk = ForwardKinematics(debug=False)
        self.init_pulse = init_pulse if init_pulse is not None else [500, 850, 200, 120, 500]
        self.init_run_time_ms = init_run_time_ms
        self.current_pulse = list(self.init_pulse)
        self.enable_pick = enable_pick
        self.pick_pitch = pick_pitch
        self.pick_cooldown_s = pick_cooldown_s
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.sync_slop = sync_slop
        self.preview_port = preview_port
        self.preview_quality = preview_quality
        self.conf_threshold = conf_threshold
        self.target_offset = np.array(target_offset if target_offset is not None else [0.0, 0.0, 0.0], dtype=float)
        self.min_camera_depth = min_camera_depth
        self.max_camera_depth = max_camera_depth
        self.depth_roi_size = max(1, int(depth_roi_size))
        self.detect_every_n = max(1, int(detect_every_n))
        self.lock_frames = max(1, int(lock_frames))
        self.lost_timeout = lost_timeout
        self.single_shot = single_shot
        self.approach_only = approach_only
        self.coord_log_interval = coord_log_interval
        self.max_pulse_step = max(1, int(max_pulse_step))
        self.move_time_per_100_pulse = max(1, int(move_time_per_100_pulse))
        self.min_move_time_ms = max(1, int(min_move_time_ms))
        self.approach_height = approach_height
        self.joint_groups = joint_groups if joint_groups is not None else [[1], [5], [2], [3], [4]]
        self.joint_sequential = joint_sequential
        self.joint_order = joint_order if joint_order is not None else [1, 5, 2, 3, 4]
        self.cartesian_motion = cartesian_motion
        self.cartesian_steps = max(1, int(cartesian_steps))
        self.direct_target = direct_target
        self.gripper_open_pulse = max(0, min(1000, int(gripper_open_pulse)))
        self.gripper_close_pulse = max(0, min(1000, int(gripper_close_pulse)))
        self.post_grip_servo1_delta = int(post_grip_servo1_delta)
        self.post_grip_time_ms = max(1, int(post_grip_time_ms))
        self.camera_ee_offset = np.array(
            camera_ee_offset if camera_ee_offset is not None else [0.05, 0.05, 0.1],
            dtype=float
        )
        self.freeze_after_lock = freeze_after_lock
        self.reset_after_motion = reset_after_motion
        self.preview_lock = threading.Lock()
        self.latest_preview_jpeg = None
        self.latest_color_seq = 0
        self.last_ai_seq = -1
        self.last_pick_time = 0.0
        self.pick_lock = threading.Lock()
        self.target_history = []
        self.locked_target_xyz = None
        self.locked_camera_xyz = None
        self.locked_box = None
        self.locked_conf = 0.0
        self.locked_target_time = 0.0
        self.locked_executed = False
        self.executing_motion = False
        self.prev_world_xyz = None
        self.last_delta_log_time = 0.0
        self.last_coord_log_time = 0.0
        self.color_msg_count = 0
        self.depth_msg_count = 0
        self.sync_msg_count = 0
        self.last_status_color_count = 0
        self.last_status_depth_count = 0
        self.last_status_sync_count = 0

        self.get_logger().info(f'机械臂初始脉冲: {self.init_pulse}')

        if not skip_init:
            self.move_to_init_pose(run_time_ms=self.init_run_time_ms)
        else:
            self.get_logger().info('已跳过初始归位。')

        time.sleep(2)
        self.get_logger().info('初始化等待结束。')

        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best.onnx')
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.ai_box = None
        self.ai_conf = 0.0

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.model_h = self.session.get_inputs()[0].shape[2]
            self.model_w = self.session.get_inputs()[0].shape[3]
            self.ai_thread = threading.Thread(target=self.ai_loop, daemon=True)
            self.ai_thread.start()
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')
            self.session = None

        self.get_logger().info(
            f'订阅相机话题 color={self.color_topic}, depth={self.depth_topic}, sync_slop={self.sync_slop}s'
        )
        self.color_subscription = self.create_subscription(
            Image, self.color_topic, self.color_callback, qos_profile_sensor_data
        )
        self.depth_subscription = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.create_timer(2.0, self.status_timer_callback)

        self.image_pub = self.create_publisher(Image, '/cucumber_picker/image_result', 10)
        if self.preview_port > 0:
            self.start_preview_server(self.preview_port)

    def start_preview_server(self, port):
        node = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    body = (
                        '<!doctype html><html><head><meta charset="utf-8">'
                        '<title>Cucumber Picker Preview</title>'
                        '<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;}'
                        'header{padding:10px 14px;background:#222;}'
                        'img{display:block;max-width:100vw;max-height:calc(100vh - 44px);margin:auto;}'
                        '</style></head><body><header>/cucumber_picker/image_result</header>'
                        '<img src="/stream.mjpg"></body></html>'
                    ).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != '/stream.mjpg':
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header('Age', '0')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()

                while True:
                    with node.preview_lock:
                        frame = node.latest_preview_jpeg
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode('ascii'))
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        break

        server = ThreadingHTTPServer(('0.0.0.0', port), PreviewHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info(f'预览地址: http://<树莓派IP>:{port}/')

    @staticmethod
    def stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def color_callback(self, msg):
        self.color_msg_count += 1
        with self.lock:
            self.latest_color_msg = msg
            depth_msg = self.latest_depth_msg
        if depth_msg is not None:
            self.try_sync_callback(msg, depth_msg)

    def depth_callback(self, msg):
        self.depth_msg_count += 1
        with self.lock:
            self.latest_depth_msg = msg
            color_msg = self.latest_color_msg
        if color_msg is not None:
            self.try_sync_callback(color_msg, msg)

    def try_sync_callback(self, color_msg, depth_msg):
        color_t = self.stamp_to_sec(color_msg.header.stamp)
        depth_t = self.stamp_to_sec(depth_msg.header.stamp)
        if abs(color_t - depth_t) > self.sync_slop:
            return
        self.sync_callback(color_msg, depth_msg)

    def get_depth_mm(self, depth_image, cx, cy):
        half = self.depth_roi_size // 2
        y1 = max(0, cy - half)
        y2 = min(depth_image.shape[0], cy + half + 1)
        x1 = max(0, cx - half)
        x2 = min(depth_image.shape[1], cx + half + 1)
        roi = depth_image[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size == 0:
            return 0
        return int(np.median(valid))

    def handle_valid_target(self, world_xyz, box=None, conf=0.0, camera_xyz=None):
        if self.executing_motion or (self.freeze_after_lock and self.locked_target_xyz is not None):
            return
        world_xyz = np.array(world_xyz, dtype=float)
        self.target_history.append(world_xyz)
        if len(self.target_history) > self.lock_frames:
            self.target_history = self.target_history[-self.lock_frames:]

        if len(self.target_history) < self.lock_frames:
            self.get_logger().info(f'目标稳定中: {len(self.target_history)}/{self.lock_frames}')
            return

        locked = np.median(np.array(self.target_history), axis=0)
        if self.locked_target_xyz is None or np.linalg.norm(locked - self.locked_target_xyz) > 0.01:
            self.locked_target_xyz = locked
            self.locked_camera_xyz = list(camera_xyz) if camera_xyz is not None else None
            self.locked_box = list(box) if box is not None else None
            self.locked_conf = float(conf)
            self.locked_target_time = time.time()
            self.locked_executed = False
            self.get_logger().info(
                f'锁定目标: X={locked[0]:.3f} Y={locked[1]:.3f} Z={locked[2]:.3f}'
            )

        if self.single_shot and self.locked_executed:
            return

        if self.locked_target_xyz is not None:
            self.locked_executed = True
            threading.Thread(
                target=self.execute_pick,
                args=(tuple(self.locked_target_xyz.tolist()),),
                daemon=True
            ).start()

    def reset_locked_target(self):
        self.target_history = []
        self.locked_target_xyz = None
        self.locked_camera_xyz = None
        self.locked_box = None
        self.locked_conf = 0.0
        self.locked_target_time = 0.0
        self.locked_executed = False
        self.prev_world_xyz = None
        with self.lock:
            self.ai_box = None
            self.ai_conf = 0.0

    def handle_lost_target(self):
        if self.executing_motion:
            return
        if self.locked_target_xyz is None:
            self.target_history = []
            return
        if time.time() - self.locked_target_time > self.lost_timeout:
            self.get_logger().info('目标丢失超时，清除锁定目标。')
            self.locked_target_xyz = None
            self.locked_camera_xyz = None
            self.locked_box = None
            self.locked_conf = 0.0
            self.target_history = []
            self.locked_executed = False

    def status_timer_callback(self):
        color_delta = self.color_msg_count - self.last_status_color_count
        depth_delta = self.depth_msg_count - self.last_status_depth_count
        sync_delta = self.sync_msg_count - self.last_status_sync_count
        self.last_status_color_count = self.color_msg_count
        self.last_status_depth_count = self.depth_msg_count
        self.last_status_sync_count = self.sync_msg_count
        self.get_logger().info(
            f'📡 相机状态(近2秒): color={color_delta}, depth={depth_delta}, sync={sync_delta}, '
            f'ai_conf={self.ai_conf:.2f}, ai_box={self.ai_box}, '
            f'locked={None if self.locked_target_xyz is None else np.round(self.locked_target_xyz, 3).tolist()}'
        )

    def choose_ik_pulse(self, position):
        radial = float(np.linalg.norm([position[0], position[1]]))
        distance = float(np.linalg.norm(position))
        all_solutions = get_ik([float(i) for i in position],
                               float(self.pick_pitch),
                               [-180.0, 180.0],
                               1.0)
        if not all_solutions:
            return None

        best_pulse = None
        best_delta = float('inf')
        for solution in all_solutions:
            for pulse in transform.angle2pulse(solution[0], True):
                pulse = [max(0, min(1000, int(i))) for i in pulse]
                delta = sum(abs(pulse[i] - self.current_pulse[i]) for i in range(5))
                if delta < best_delta:
                    best_delta = delta
                    best_pulse = pulse

        return best_pulse

    def move_grouped_servos(self, pulse, use_time):
        for group in self.joint_groups:
            active = []
            for sid in group:
                if 1 <= sid <= 5:
                    value = int(pulse[sid - 1])
                    active.append((sid, value))
            if active:
                sc.setServoPulses(active, use_time)
                self.get_logger().info(f'同包分组移动 {active}, time={use_time}ms')
                time.sleep(use_time / 1000.0 + 0.15)

    def move_joint_sequential(self, pulse):
        target = [max(0, min(1000, int(value))) for value in pulse]
        self.current_pulse = [max(0, min(1000, int(value))) for value in self.current_pulse]

        for sid in self.joint_order:
            if not 1 <= sid <= 5:
                continue
            idx = sid - 1
            start = int(self.current_pulse[idx])
            end = int(target[idx])
            delta = end - start
            steps = max(1, int(np.ceil(abs(delta) / self.max_pulse_step)))
            self.get_logger().info(f'单关节移动 joint {sid}: {start} -> {end}, steps={steps}')

            for step in range(1, steps + 1):
                ratio = step / steps
                value = int(round(start + delta * ratio))
                waypoint_delta = abs(value - self.current_pulse[idx])
                use_time = int(max(self.min_move_time_ms, waypoint_delta * self.move_time_per_100_pulse / 100.0))
                self.current_pulse[idx] = value
                self.get_logger().info(
                    f'单关节 step {step}/{steps}: joint={sid}, pulse={self.current_pulse}, time={use_time}ms'
                )
                sc.setServoPulse(sid, value, use_time)
                time.sleep(use_time / 1000.0 + 0.2)
                self.log_end_effector_pose(self.current_pulse, f'joint {sid} step {step}/{steps}')

    def log_end_effector_pose(self, pulse, label='当前'):
        try:
            angles = transform.pulse2angle(pulse)
            fk_res = self.fk.get_fk(angles)
            xyz = fk_res[0]
            self.get_logger().info(
                f'末端坐标({label}) -> X_base:{xyz[0]:.3f}m Y_base:{xyz[1]:.3f}m Z_base:{xyz[2]:.3f}m pulse={pulse}'
            )
        except Exception as e:
            self.get_logger().warn(f'末端坐标计算失败: {e}')

    def get_end_effector_xyz(self, pulse=None):
        pulse = self.current_pulse if pulse is None else pulse
        angles = transform.pulse2angle(pulse)
        fk_res = self.fk.get_fk(angles)
        return np.array(fk_res[0], dtype=float)

    def move_cartesian_to(self, target_xyz, label='cartesian'):
        start_xyz = self.get_end_effector_xyz()
        target_xyz = np.array(target_xyz, dtype=float)
        self.get_logger().info(
            f'笛卡尔移动({label}): '
            f'from=({start_xyz[0]:.3f}, {start_xyz[1]:.3f}, {start_xyz[2]:.3f}) '
            f'to=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}), '
            f'steps={self.cartesian_steps}'
        )

        for step in range(1, self.cartesian_steps + 1):
            ratio = step / self.cartesian_steps
            xyz = start_xyz + (target_xyz - start_xyz) * ratio
            pulse = self.choose_ik_pulse(xyz)
            if pulse is None:
                r_xy = float(np.linalg.norm(xyz[:2]))
                self.get_logger().warn(
                    f'笛卡尔 step {step}/{self.cartesian_steps} IK无解: '
                    f'xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}), r={r_xy:.3f}'
                )
                return False

            self.get_logger().info(
                f'笛卡尔 step {step}/{self.cartesian_steps}: '
                f'xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) pulse={pulse}'
            )
            self.move_arm(pulse)

        return True

    def set_gripper(self, pulse, label, use_time=700, repeats=2):
        raw_pulse = 1000 - int(pulse)
        self.get_logger().info(
            f'{label}夹爪: servo10 api_pulse={int(pulse)}, raw_pulse={raw_pulse}, repeats={repeats}'
        )
        for _ in range(max(1, int(repeats))):
            sc.setServoPulse(10, int(pulse), int(use_time))
            time.sleep(use_time / 1000.0 + 0.15)

    def twist_servo1_and_release(self):
        start = int(self.current_pulse[0])
        target = max(0, min(1000, start + self.post_grip_servo1_delta))
        if target == start:
            self.get_logger().warn(
                f'1号舵机旋转跳过: start={start}, delta={self.post_grip_servo1_delta} 超出范围'
            )
        else:
            self.get_logger().info(
                f'夹取后旋转1号舵机: {start} -> {target}, '
                f'delta={self.post_grip_servo1_delta}, time={self.post_grip_time_ms}ms'
            )
            sc.setServoPulse(1, target, self.post_grip_time_ms)
            time.sleep(self.post_grip_time_ms / 1000.0 + 0.2)
            self.current_pulse[0] = target
            self.log_end_effector_pose(self.current_pulse, 'servo1旋转后')

        self.set_gripper(self.gripper_open_pulse, '旋转后松开', use_time=700, repeats=2)

    def move_arm(self, pulse, run_time_ms=None):
        if self.joint_sequential and run_time_ms is None:
            self.move_joint_sequential(pulse)
            return

        target = [max(0, min(1000, int(value))) for value in pulse]
        current = [int(value) for value in self.current_pulse]
        max_delta = max(abs(target[i] - current[i]) for i in range(5))
        steps = max(1, int(np.ceil(max_delta / self.max_pulse_step)))

        for step in range(1, steps + 1):
            ratio = step / steps
            waypoint = [
                int(round(current[i] + (target[i] - current[i]) * ratio))
                for i in range(5)
            ]
            waypoint_delta = max(abs(waypoint[i] - self.current_pulse[i]) for i in range(5))
            auto_time = int(max(self.min_move_time_ms, waypoint_delta * self.move_time_per_100_pulse / 100.0))
            use_time = int(run_time_ms) if run_time_ms is not None else auto_time
            self.get_logger().info(f'缓慢移动 step {step}/{steps}: {waypoint}, time={use_time}ms')
            self.move_grouped_servos(waypoint, use_time)
            self.current_pulse = list(waypoint)
            self.log_end_effector_pose(self.current_pulse, f'step {step}/{steps}')

    def execute_pick(self, target_xyz):
        if not self.pick_lock.acquire(blocking=False):
            return
        self.executing_motion = True

        try:
            now = time.time()
            if now - self.last_pick_time < self.pick_cooldown_s:
                return
            self.last_pick_time = now

            target = np.array(target_xyz, dtype=float) + self.target_offset
            approach = target.copy()
            approach[2] += self.approach_height

            self.get_logger().info(
                f'执行锁定目标: target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}), '
                f'approach=({approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f})'
            )

            if self.enable_pick:
                self.set_gripper(self.gripper_open_pulse, '锁定后打开', use_time=700, repeats=2)

            approach_pulse = self.choose_ik_pulse(approach)
            target_pulse = self.choose_ik_pulse(target)

            if approach_pulse is None or target_pulse is None:
                approach_r = float(np.linalg.norm(approach[:2]))
                target_r = float(np.linalg.norm(target[:2]))
                self.get_logger().warn(
                    'IK 无解，跳过采摘: '
                    f'approach=({approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}, r={approach_r:.3f}), '
                    f'target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}, r={target_r:.3f}), '
                    f'offset={self.target_offset.tolist()}, pitch={self.pick_pitch:.1f}'
                )
                return

            self.get_logger().info(f'IK approach pulse: {approach_pulse}')
            self.get_logger().info(f'IK target pulse: {target_pulse}')
            self.log_end_effector_pose(approach_pulse, 'approach目标')
            self.log_end_effector_pose(target_pulse, 'target目标')

            if not self.enable_pick:
                self.get_logger().info('未启用 --enable-pick，仅输出 IK，不发送采摘动作。')
                return

            if self.direct_target:
                self.get_logger().info('direct-target 模式：锁定后直接移动到目标点。')
                if self.cartesian_motion:
                    if not self.move_cartesian_to(target, 'direct-target'):
                        return
                else:
                    self.move_arm(target_pulse)
                self.log_end_effector_pose(self.current_pulse, 'direct-target到位')
                self.set_gripper(self.gripper_close_pulse, '目标到位后闭合', use_time=700, repeats=2)
                self.twist_servo1_and_release()
                self.get_logger().info('本轮动作结束，恢复初始姿态。')
                self.move_arm(self.init_pulse)
                return

            if self.cartesian_motion:
                if not self.move_cartesian_to(approach, 'approach'):
                    return
            else:
                self.move_arm(approach_pulse)
            if self.approach_only:
                self.get_logger().info('approach-only 模式：已移动到目标上方，未下探和夹取。')
                return
            if self.cartesian_motion:
                if not self.move_cartesian_to(target, 'target'):
                    return
            else:
                self.move_arm(target_pulse)
            self.set_gripper(self.gripper_close_pulse, '目标到位后闭合', use_time=700, repeats=2)
            self.twist_servo1_and_release()
            if self.cartesian_motion:
                self.move_cartesian_to(approach, 'lift')
            else:
                self.move_arm(approach_pulse)
            self.move_arm(self.init_pulse)
        finally:
            self.executing_motion = False
            if self.reset_after_motion:
                self.get_logger().info('本轮动作结束，清空锁定目标，等待下一次识别。')
                self.reset_locked_target()
            self.pick_lock.release()

    def move_to_init_pose(self, run_time_ms=1500):
        self.get_logger().info(f'发送初始脉冲: {self.init_pulse}，耗时 {run_time_ms}ms')
        self.move_grouped_servos(self.init_pulse, run_time_ms)
        self.current_pulse = list(self.init_pulse)
        self.log_end_effector_pose(self.current_pulse, 'init')
        # 夹爪保持打开
        self.set_gripper(self.gripper_open_pulse, '初始化打开', use_time=run_time_ms, repeats=1)

    # 推理线程只处理新图像，按配置间隔执行检测。
    def ai_loop(self):
        while True:
            color_img = None
            color_seq = -1
            with self.lock:
                if self.latest_color is not None:
                    color_img = self.latest_color.copy()
                    color_seq = self.latest_color_seq
            if color_img is None:
                time.sleep(0.05)
                continue
            if color_seq == self.last_ai_seq or color_seq % self.detect_every_n != 0:
                time.sleep(0.01)
                continue
            self.last_ai_seq = color_seq

            img_h, img_w = color_img.shape[:2]
            input_img = cv2.resize(color_img, (self.model_w, self.model_h))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = np.expand_dims(input_img, axis=0)

            outputs = self.session.run(None, {self.input_name: input_img})
            preds = outputs[0][0].T
            best_conf = 0.0
            best_box = None

            for pred in preds:
                box = pred[:4]
                confs = pred[4:]
                class_id = np.argmax(confs)
                conf = confs[class_id]
                if conf > self.conf_threshold and conf > best_conf:
                    best_conf = conf
                    best_box = box

            with self.lock:
                self.ai_conf = best_conf
                if best_box is not None:
                    x_center, y_center, width, height = best_box
                    self.ai_box = [
                        int(x_center / self.model_w * img_w),
                        int(y_center / self.model_h * img_h),
                        int(width / self.model_w * img_w),
                        int(height / self.model_h * img_h)
                    ]
                else:
                    self.ai_box = None

    # 对齐深度与检测框配合完成定位和结果发布。
    def sync_callback(self, color_msg, depth_msg):
        self.sync_msg_count += 1
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        img_h, img_w = color_image.shape[:2]

        with self.lock:
            self.latest_color = color_image
            self.latest_depth = depth_image
            self.latest_color_seq += 1
            box = self.ai_box
            conf = self.ai_conf

        if self.freeze_after_lock and self.locked_target_xyz is not None:
            locked = self.locked_target_xyz
            locked_camera = self.locked_camera_xyz
            locked_box = self.locked_box
            cv2.putText(
                color_image,
                f"LOCKED BASE X:{locked[0]:.3f} Y:{locked[1]:.3f} Z:{locked[2]:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            if locked_camera is not None:
                cv2.putText(
                    color_image,
                    f"CAM X:{locked_camera[0]:.3f} Y:{locked_camera[1]:.3f} Z:{locked_camera[2]:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2
                )
            if locked_box is not None:
                lx, ly, lw, lh = locked_box
                lx1, ly1 = int(lx - lw / 2), int(ly - lh / 2)
                lx2, ly2 = int(lx + lw / 2), int(ly + lh / 2)
                cv2.rectangle(color_image, (lx1, ly1), (lx2, ly2), (255, 180, 0), 3)
                cv2.putText(
                    color_image,
                    f"LOCKED cucumber {self.locked_conf:.2f}",
                    (lx1, max(25, ly1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 180, 0),
                    2
                )
            box = None

        if box is None:
            self.handle_lost_target()
        else:
            cx, cy, width, height = box
            if 0 <= cx < img_w and 0 <= cy < img_h:
                depth_mm = self.get_depth_mm(depth_image, cx, cy)
                if depth_mm > 0:
                    depth_m = float(depth_mm) / 1000.0
                    x1, y1 = int(cx - width / 2), int(cy - height / 2)
                    x2, y2 = int(cx + width / 2), int(cy + height / 2)

                    if depth_m < self.min_camera_depth or depth_m > self.max_camera_depth:
                        self.get_logger().warn(
                            f'忽略目标: 深度 {depth_m:.3f}m 超出范围 '
                            f'[{self.min_camera_depth:.3f}, {self.max_camera_depth:.3f}]m, '
                            f'conf={conf:.2f}, box={box}'
                        )
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(color_image, f"ignored {conf:.2f} {depth_mm}mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        # 内参对应当前设备的 640x480 彩色流，更换设备后需重新读取。
                        fx = 387.1556701660156
                        fy = 387.1556701660156
                        cx_param = 320.6669921875
                        cy_param = 236.83148193359375

                        X_camera = (cx - cx_param) * float(depth_mm) / fx / 1000.0
                        Y_camera = (cy - cy_param) * float(depth_mm) / fy / 1000.0
                        Z_camera = depth_m

                        end_effector_xyz3 = np.array(
                            [Z_camera, -X_camera, -Y_camera],
                            dtype=float
                        ) + self.camera_ee_offset
                        end_effector_xyz = np.array([
                            end_effector_xyz3[0],
                            end_effector_xyz3[1],
                            end_effector_xyz3[2],
                            1.0
                        ])

                        angles = transform.pulse2angle(self.current_pulse)
                        fk_res = self.fk.get_fk(angles)
                        cur_xyz = fk_res[0]
                        cur_quat = fk_res[1]

                        endpoint = np.eye(4)
                        endpoint[:3, :3] = quaternion_to_rotation_matrix(cur_quat)
                        endpoint[:3, 3] = cur_xyz

                        world_target_xyz = endpoint.dot(end_effector_xyz)
                        X_world, Y_world, Z_world = world_target_xyz[:3]

                        now = time.time()
                        if now - self.last_coord_log_time > self.coord_log_interval:
                            self.get_logger().info(
                                f'相机坐标 -> X:{X_camera:.3f}m Y:{Y_camera:.3f}m Z:{Z_camera:.3f}m \n'
                                f'末端坐标 -> X_ee:{end_effector_xyz3[0]:.3f}m Y_ee:{end_effector_xyz3[1]:.3f}m Z_ee:{end_effector_xyz3[2]:.3f}m, '
                                f'fixed_map=[cam_z,-cam_x,-cam_y], offset={self.camera_ee_offset.tolist()}\n'
                                f'基座坐标 -> X_base:{X_world:.3f}m Y_base:{Y_world:.3f}m Z_base:{Z_world:.3f}m, '
                                f'current_pulse={self.current_pulse}'
                            )
                            self.last_coord_log_time = now

                        if self.prev_world_xyz is None:
                            self.prev_world_xyz = (X_world, Y_world, Z_world)
                            self.last_delta_log_time = now
                        elif (now - self.last_delta_log_time) > 0.5:
                            d_x = X_world - self.prev_world_xyz[0]
                            d_y = Y_world - self.prev_world_xyz[1]
                            d_z = Z_world - self.prev_world_xyz[2]
                            if abs(d_x) > 0.001 or abs(d_y) > 0.001 or abs(d_z) > 0.001:
                                self.get_logger().info(
                                    f'坐标变化 -> dX:{d_x:+.3f}m dY:{d_y:+.3f}m dZ:{d_z:+.3f}m'
                                )

                            self.prev_world_xyz = (X_world, Y_world, Z_world)
                            self.last_delta_log_time = now

                        self.handle_valid_target(
                            (X_world, Y_world, Z_world),
                            box,
                            conf,
                            (X_camera, Y_camera, Z_camera)
                        )

                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.circle(color_image, (cx, cy), 8, (0, 255, 0), -1)
                        cv2.putText(color_image, f"{conf:.2f} {depth_mm}mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 结果通过 ROS2 和网页预览输出，支持无桌面环境。
        try:
            result_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            result_msg.header = color_msg.header # 保持相同时间戳
            self.image_pub.publish(result_msg)
            if self.preview_port > 0:
                ok, jpg = cv2.imencode(
                    '.jpg',
                    color_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.preview_quality)]
                )
                if ok:
                    with self.preview_lock:
                        self.latest_preview_jpeg = jpg.tobytes()
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--init-pulse', type=str,
                        default=os.getenv('INIT_PULSE', '500,850,200,120,500'),
                        help='初始舵机pulse列表, 用逗号分隔。例如 500,850,200,120,500')
    parser.add_argument('--init-run-time-ms', type=int,
                        default=int(os.getenv('INIT_RUN_TIME_MS', '1500')),
                        help='初始舵机归位时间，单位毫秒')
    parser.add_argument('--skip-init', action='store_true',
                        help='启动时不执行初始舵机归位，仅加载节点')
    parser.add_argument('--enable-pick', action='store_true',
                        help='启用真实采摘动作；不加该参数时只计算和打印IK结果')
    parser.add_argument('--pick-pitch', type=float,
                        default=float(os.getenv('PICK_PITCH', '0.0')),
                        help='末端采摘俯仰角，单位度')
    parser.add_argument('--pick-cooldown-s', type=float,
                        default=float(os.getenv('PICK_COOLDOWN_S', '6.0')),
                        help='两次采摘动作之间的最小间隔，单位秒')
    parser.add_argument('--color-topic', type=str,
                        default=os.getenv('COLOR_TOPIC', '/camera/camera/color/image_raw'),
                        help='彩色图像话题')
    parser.add_argument('--depth-topic', type=str,
                        default=os.getenv('DEPTH_TOPIC', '/camera/camera/aligned_depth_to_color/image_raw'),
                        help='对齐到彩色图的深度图像话题')
    parser.add_argument('--sync-slop', type=float,
                        default=float(os.getenv('SYNC_SLOP', '0.3')),
                        help='彩色图和深度图近似同步容差，单位秒')
    parser.add_argument('--preview-port', type=int,
                        default=int(os.getenv('PREVIEW_PORT', '8080')),
                        help='MJPEG实时预览端口；设为0可关闭')
    parser.add_argument('--preview-quality', type=int,
                        default=int(os.getenv('PREVIEW_QUALITY', '80')),
                        help='MJPEG预览JPEG质量，1-100')
    parser.add_argument('--conf-threshold', type=float,
                        default=float(os.getenv('CONF_THRESHOLD', '0.75')),
                        help='检测置信度阈值，调高可减少误检')
    parser.add_argument('--target-offset', type=str,
                        default=os.getenv('TARGET_OFFSET', '0,0,0'),
                        help='底座目标坐标修正量x,y,z，单位m。例如 0,0,-0.10')
    parser.add_argument('--camera-ee-offset', type=str,
                        default=os.getenv('CAMERA_EE_OFFSET', '0.05,0.05,0.1'),
                        help='相机到末端坐标系的平移补偿x,y,z，单位m。例如 0.05,0.05,0.1')
    parser.add_argument('--min-camera-depth', type=float,
                        default=float(os.getenv('MIN_CAMERA_DEPTH', '0.08')),
                        help='接受的最小相机深度，单位m')
    parser.add_argument('--max-camera-depth', type=float,
                        default=float(os.getenv('MAX_CAMERA_DEPTH', '0.80')),
                        help='接受的最大相机深度，单位m；远处误检会被忽略')
    parser.add_argument('--depth-roi-size', type=int,
                        default=int(os.getenv('DEPTH_ROI_SIZE', '9')),
                        help='检测框中心取深度的中位数窗口大小，单位像素')
    parser.add_argument('--detect-every-n', type=int,
                        default=int(os.getenv('DETECT_EVERY_N', '3')),
                        help='每N帧运行一次ONNX检测，降低树莓派CPU压力')
    parser.add_argument('--lock-frames', type=int,
                        default=int(os.getenv('LOCK_FRAMES', '5')),
                        help='连续收集多少个有效目标后用中位数锁定目标')
    parser.add_argument('--lost-timeout', type=float,
                        default=float(os.getenv('LOST_TIMEOUT', '2.0')),
                        help='锁定后目标丢失多久才清除锁定，单位秒')
    parser.add_argument('--single-shot', action='store_true',
                        help='锁定目标后只触发一次IK/动作，适合调试')
    parser.add_argument('--approach-only', action='store_true',
                        help='启用真实动作时只移动到目标上方，不下探不夹取')
    parser.add_argument('--coord-log-interval', type=float,
                        default=float(os.getenv('COORD_LOG_INTERVAL', '1.0')),
                        help='坐标日志输出间隔，单位秒')
    parser.add_argument('--max-pulse-step', type=int,
                        default=int(os.getenv('MAX_PULSE_STEP', '40')),
                        help='机械臂分段移动时每步最大脉宽变化，越小越慢越稳')
    parser.add_argument('--move-time-per-100-pulse', type=int,
                        default=int(os.getenv('MOVE_TIME_PER_100_PULSE', '1000')),
                        help='每100脉宽变化对应的移动时间，单位ms')
    parser.add_argument('--min-move-time-ms', type=int,
                        default=int(os.getenv('MIN_MOVE_TIME_MS', '1200')),
                        help='分段移动每一步的最小移动时间，单位ms')
    parser.add_argument('--approach-height', type=float,
                        default=float(os.getenv('APPROACH_HEIGHT', '0.05')),
                        help='approach点相对目标点的高度偏移，单位m')
    parser.add_argument('--joint-groups', type=str,
                        default=os.getenv('JOINT_GROUPS', '1;5;2;3;4'),
                        help='舵机分组移动顺序，例如 "1;5;2;3;4" 表示全部单独移动')
    parser.add_argument('--joint-sequential', action='store_true',
                        help='每个关节分步到目标脉冲后，再移动下一个关节')
    parser.add_argument('--joint-order', type=str,
                        default=os.getenv('JOINT_ORDER', '1,5,2,3,4'),
                        help='--joint-sequential 的关节移动顺序，例如 1,5,2,3,4')
    parser.add_argument('--cartesian-motion', action='store_true',
                        help='使用基座坐标系直线插值路径，每个中间xyz重新IK')
    parser.add_argument('--cartesian-steps', type=int,
                        default=int(os.getenv('CARTESIAN_STEPS', '6')),
                        help='--cartesian-motion 的空间插值步数')
    parser.add_argument('--direct-target', action='store_true',
                        help='锁定后直接移动到target点，跳过approach/lift/init动作')
    parser.add_argument('--gripper-open-pulse', type=int,
                        default=int(os.getenv('GRIPPER_OPEN_PULSE', '500')),
                        help='夹爪打开pulse；如果夹爪半闭合，可尝试调大或调小')
    parser.add_argument('--gripper-close-pulse', type=int,
                        default=int(os.getenv('GRIPPER_CLOSE_PULSE', '590')),
                        help='夹爪闭合pulse；数值越靠近打开pulse通常夹得越松')
    parser.add_argument('--post-grip-servo1-delta', type=int,
                        default=int(os.getenv('POST_GRIP_SERVO1_DELTA', '375')),
                        help='夹爪闭合后1号舵机旋转的pulse增量；90度约375，方向反了用-375')
    parser.add_argument('--post-grip-time-ms', type=int,
                        default=int(os.getenv('POST_GRIP_TIME_MS', '1500')),
                        help='夹爪闭合后1号舵机旋转耗时，单位ms')
    parser.add_argument('--no-freeze-after-lock', action='store_true',
                        help='锁定目标后仍继续处理新识别，默认锁定后冻结视觉控制')
    parser.add_argument('--keep-locked-after-motion', action='store_true',
                        help='动作结束后不清空锁定目标，默认清空后等待下一次识别')
    args, unknown = parser.parse_known_args(argv)

    pulse_values = [int(x) for x in args.init_pulse.split(',') if x.strip()]
    target_offset = [float(x) for x in args.target_offset.split(',') if x.strip()]
    if len(target_offset) != 3:
        raise ValueError('--target-offset 必须包含3个数，例如 0,0,-0.10')
    camera_ee_offset = [float(x) for x in args.camera_ee_offset.split(',') if x.strip()]
    if len(camera_ee_offset) != 3:
        raise ValueError('--camera-ee-offset 必须包含3个数，例如 -0.101,0,0.037')
    joint_groups = []
    for group in args.joint_groups.split(';'):
        ids = [int(x) for x in group.split(',') if x.strip()]
        if ids:
            joint_groups.append(ids)
    if not joint_groups:
        raise ValueError('--joint-groups 至少包含一个舵机分组，例如 "1,5;2,3;4"')
    joint_order = [int(x) for x in args.joint_order.split(',') if x.strip()]
    if not joint_order:
        raise ValueError('--joint-order 至少包含一个关节编号，例如 1,5,2,3,4')
    return args, pulse_values, target_offset, camera_ee_offset, joint_groups, joint_order, unknown


def main(args=None):
    parsed_args, pulse_values, target_offset, camera_ee_offset, joint_groups, joint_order, rclpy_args = parse_args(args)
    rclpy.init(args=rclpy_args)
    node = CucumberPickerNode(init_pulse=pulse_values,
                               init_run_time_ms=parsed_args.init_run_time_ms,
                               skip_init=parsed_args.skip_init,
                               enable_pick=parsed_args.enable_pick,
                               pick_pitch=parsed_args.pick_pitch,
                               pick_cooldown_s=parsed_args.pick_cooldown_s,
                               color_topic=parsed_args.color_topic,
                               depth_topic=parsed_args.depth_topic,
                               sync_slop=parsed_args.sync_slop,
                               preview_port=parsed_args.preview_port,
                               preview_quality=parsed_args.preview_quality,
                               conf_threshold=parsed_args.conf_threshold,
                               target_offset=target_offset,
                               camera_ee_offset=camera_ee_offset,
                               min_camera_depth=parsed_args.min_camera_depth,
                               max_camera_depth=parsed_args.max_camera_depth,
                               depth_roi_size=parsed_args.depth_roi_size,
                               detect_every_n=parsed_args.detect_every_n,
                               lock_frames=parsed_args.lock_frames,
                               lost_timeout=parsed_args.lost_timeout,
                               single_shot=parsed_args.single_shot,
                               approach_only=parsed_args.approach_only,
                               coord_log_interval=parsed_args.coord_log_interval,
                               max_pulse_step=parsed_args.max_pulse_step,
                               move_time_per_100_pulse=parsed_args.move_time_per_100_pulse,
                               min_move_time_ms=parsed_args.min_move_time_ms,
                               approach_height=parsed_args.approach_height,
                               joint_groups=joint_groups,
                               joint_sequential=parsed_args.joint_sequential,
                               joint_order=joint_order,
                               cartesian_motion=parsed_args.cartesian_motion,
                               cartesian_steps=parsed_args.cartesian_steps,
                               direct_target=parsed_args.direct_target,
                               gripper_open_pulse=parsed_args.gripper_open_pulse,
                               gripper_close_pulse=parsed_args.gripper_close_pulse,
                               post_grip_servo1_delta=parsed_args.post_grip_servo1_delta,
                               post_grip_time_ms=parsed_args.post_grip_time_ms,
                               freeze_after_lock=not parsed_args.no_freeze_after_lock,
                               reset_after_motion=not parsed_args.keep_locked_after_motion)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
