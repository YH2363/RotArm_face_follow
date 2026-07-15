import time
import threading
from collections import deque
from pathlib import Path
from statistics import median
from time import sleep

import cv2 as cv
import numpy as np
import ipywidgets as widgets
from IPython.display import display


# ==================== 集中配置 ====================

# --- 摄像头 ---
CAMERA_INDEX = None          # 自动探测
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_FPS   = 30
FLIP_IMAGE   = True

# --- 机械臂（与参考文件一致） ---
INITIAL_POSE      = [90, 135, 25, 25, 90, 30]
MOVE_TIME_MS      = 800
PAN_SERVO_ID      = 1
TILT_SERVO_ID     = 2
TILT_AUX_SERVO_ID = 3
PAN_LIMITS        = (10.0, 170.0)
TILT_LIMITS       = (30.0, 150.0)
TILT_AUX_LIMITS   = (20.0, 150.0)
PAN_DIRECTION     = -1.0
TILT_DIRECTION    =  1.0

# --- PID / 死区 ---
PAN_DEAD_ZONE    = 16
TILT_DEAD_ZONE   = 28
CONTROL_INTERVAL = 0.05           # 控制周期 50ms（降低更新频率，减少抖动）
SERVO_MOVE_TIME_MS=500   # 舵机移动时间 300ms（对齐 corlor_follow 的长过渡策略）
SERVO_SKIP_CYCLES  = 2          # 舵机跳帧：每 N 个控制周期才发一次舵机命令
PAN_KP         = 0.16
PAN_MAX_SPEED  = 24.0
PAN_ACCEL      = 200.0
TILT_KP        = 0.12
TILT_MAX_SPEED = 20.0
TILT_ACCEL     = 160.0
TILT_HANDOFF_MARGIN = 15.0

# --- 速度平滑（对齐 corlor_follow 的惯性环节） ---
VELOCITY_SMOOTHING_ALPHA = 0.35   # 低通滤波系数（越小越平滑，越大响应越快）

# --- 中心平滑 ---
PAN_SMOOTHING_ALPHA  = 0.75
TILT_SMOOTHING_ALPHA = 0.65
TARGET_MEDIAN_WINDOW = 1

# --- 人脸检测 ---
DETECTION_SCALE       = 0.5
MIN_FACE_SIZE         = (40, 40)
CASCADE_SCALE_FACTOR  = 1.2
CASCADE_MIN_NEIGHBORS = 5

# --- 显示 ---
DISPLAY_EVERY_N_FRAMES = 2
JPEG_QUALITY           = 60


# ==================== 自动探测 ====================

# Arm_Lib
try:
    import Arm_Lib
    HAS_ARM = True
    print('[OK] Arm_Lib 导入成功')
except ImportError:
    HAS_ARM = False
    print('[WARN] Arm_Lib 未找到（非 200DK 环境）')

# 摄像头
for cam_idx in range(3):
    cap_test = cv.VideoCapture(cam_idx)
    if cap_test.isOpened():
        CAMERA_INDEX = cam_idx
        cap_test.release()
        print(f'[OK] 摄像头已探测到，设备编号: {cam_idx}')
        break
if CAMERA_INDEX is None:
    print('[WARN] 未探测到任何摄像头设备')

# 人脸模型
local_model = Path('haarcascade_frontalface_default.xml')
if local_model.exists():
    FACE_MODEL_PATH = str(local_model)
else:
    FACE_MODEL_PATH = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
print(f'[OK] 人脸模型: {FACE_MODEL_PATH}')

Arm = None
arm_lock = threading.Lock()

def write_arm_pose(pose, move_time_ms):
    if Arm is None:
        return
    angles = [int(round(v)) for v in pose]
    with arm_lock:
        if hasattr(Arm, 'Arm_serial_servo_write6_array'):
            Arm.Arm_serial_servo_write6_array(angles, int(move_time_ms))
        else:
            Arm.Arm_serial_servo_write6(*angles, int(move_time_ms))

if HAS_ARM:
    Arm = Arm_Lib.Arm_Device()
    time.sleep(0.2)
    write_arm_pose(INITIAL_POSE, MOVE_TIME_MS)
    time.sleep(MOVE_TIME_MS / 1000.0)
    print(f'[OK] 机械臂已初始化到: {INITIAL_POSE}')
else:
    print('[SKIP] 跳过机械臂初始化')

class PositionPID:
    """
    时间制位置式 PID，输出解释为角速度（度/秒）。
    完全对齐 face_tracking.ipynb 中的 PositionPID。
    """

    def __init__(self, kp, ki=0.0, kd=0.0, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def update(self, error, now):
        if self.previous_time is None:
            dt, derivative = 0.0, 0.0
        else:
            dt = max(now - self.previous_time, 1e-6)
            derivative = (error - self.previous_error) / dt
        self.integral += error * dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limit is not None:
            output = float(np.clip(output, -self.output_limit, self.output_limit))
        self.previous_error = error
        self.previous_time = now
        return output
    
class FaceTrack:
    """
    融合方案（方向已修复）：
    - 舵机控制：完全对齐 face_tracking.ipynb（时间基速度 PID + 加速度限幅）
    - 人脸检测：降采样 (0.5x) + equalizeHist
    - 中心平滑：中值窗口 + EMA 指数加权
    - 帧率优化：跳帧 + JPEG 60
    """

    def __init__(self, arm=None):
        self.arm = arm
        self.face_cascade = cv.CascadeClassifier(FACE_MODEL_PATH)
        if self.face_cascade.empty():
            raise RuntimeError('Cannot load face model: ' + FACE_MODEL_PATH)

        # PID
        self.pan_pid  = PositionPID(PAN_KP,  output_limit=PAN_MAX_SPEED)
        self.tilt_pid = PositionPID(TILT_KP, output_limit=TILT_MAX_SPEED)

        # 角度
        self.pan_angle      = float(INITIAL_POSE[PAN_SERVO_ID - 1])
        self.tilt_angle     = float(INITIAL_POSE[TILT_SERVO_ID - 1])
        self.tilt_aux_angle = float(INITIAL_POSE[TILT_AUX_SERVO_ID - 1])

        # 速度
        self.pan_speed  = 0.0
        self.tilt_speed = 0.0

        # 速度低通滤波：对齐 corlor_follow 的惯性环节，平滑速度跳变
        self.filtered_pan_speed  = 0.0
        self.filtered_tilt_speed = 0.0

        self.last_control_time = None
        self.last_send_time    = 0.0
        self._servo_cycle_count = 0

        # 中心平滑
        self.center_history  = deque(maxlen=TARGET_MEDIAN_WINDOW)
        self.smoothed_center = None

    # ---------- 工具 ----------
    @staticmethod
    def clamp(value, limits):
        return max(limits[0], min(value, limits[1]))

    @staticmethod
    def slew(current, target, max_change):
        return current + np.clip(target - current, -max_change, max_change)

    @staticmethod
    def face_filter(faces):
        valid = [(int(x), int(y), int(w), int(h))
                 for x, y, w, h in faces
                 if w >= 10 and h >= 10]
        return max(valid, key=lambda f: f[2] * f[3]) if valid else None

    def reset_pid(self):
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.pan_speed  = 0.0
        self.tilt_speed = 0.0
        self.filtered_pan_speed  = 0.0
        self.filtered_tilt_speed = 0.0
        self.last_control_time = None
        self._servo_cycle_count = 0
        self.center_history.clear()
        self.smoothed_center = None

    # ---------- 舵机控制（完全对齐参考文件） ----------
    def _update_servos(self, center_x, center_y):
        now = time.monotonic()
        if now - self.last_send_time < CONTROL_INTERVAL:
            return

        dt = (CONTROL_INTERVAL if self.last_control_time is None
              else min(now - self.last_control_time, 0.1))
        self.last_control_time = now
        self.last_send_time    = now

        # 死区
        error_x = FRAME_WIDTH / 2.0 - center_x
        error_y = FRAME_HEIGHT / 2.0 - center_y
        if abs(error_x) < PAN_DEAD_ZONE:
            error_x = 0.0
        if abs(error_y) < TILT_DEAD_ZONE:
            error_y = 0.0

        # PID → 角速度
        target_pan_speed = (
            0.0 if error_x == 0.0
            else PAN_DIRECTION * self.pan_pid.update(error_x, now)
        )
        target_tilt_speed = (
            0.0 if error_y == 0.0
            else TILT_DIRECTION * self.tilt_pid.update(error_y, now)
        )

        # 速度低通滤波：对齐 corlor_follow 的惯性环节，平滑速度跳变
        alpha = VELOCITY_SMOOTHING_ALPHA
        self.filtered_pan_speed = (
            alpha * self.filtered_pan_speed
            + (1.0 - alpha) * target_pan_speed
        )
        self.filtered_tilt_speed = (
            alpha * self.filtered_tilt_speed
            + (1.0 - alpha) * target_tilt_speed
        )

        # 加速度限幅
        self.pan_speed = (
            0.0 if error_x == 0.0
            else self.slew(self.pan_speed,  self.filtered_pan_speed,  PAN_ACCEL * dt)
        )
        self.tilt_speed = (
            0.0 if error_y == 0.0
            else self.slew(self.tilt_speed, self.filtered_tilt_speed, TILT_ACCEL * dt)
        )

        # 角度积分
        pan_delta  = self.pan_speed  * dt
        tilt_delta = self.tilt_speed * dt

        self.pan_angle = self.clamp(
            self.pan_angle + pan_delta, PAN_LIMITS
        )

        # 3 号舵机优先，限位时 2 号平滑补偿
        distance = (
            (TILT_AUX_LIMITS[1] - self.tilt_aux_angle)
            if tilt_delta >= 0
            else (self.tilt_aux_angle - TILT_AUX_LIMITS[0])
        )
        aux_share = float(np.clip(
            distance / TILT_HANDOFF_MARGIN, 0.0, 1.0
        ))
        old_aux = self.tilt_aux_angle
        self.tilt_aux_angle = self.clamp(
            self.tilt_aux_angle + tilt_delta * aux_share, TILT_AUX_LIMITS
        )
        remaining = tilt_delta - (self.tilt_aux_angle - old_aux)
        self.tilt_angle = self.clamp(
            self.tilt_angle + remaining, TILT_LIMITS
        )

        # 发送命令（跳帧发送，对齐 corlor_follow 的发送节奏）
        self._servo_cycle_count += 1
        if self._servo_cycle_count % SERVO_SKIP_CYCLES == 0:
            pose = list(INITIAL_POSE)
            pose[PAN_SERVO_ID - 1]      = self.pan_angle
            pose[TILT_SERVO_ID - 1]     = self.tilt_angle
            pose[TILT_AUX_SERVO_ID - 1] = self.tilt_aux_angle

            if self.arm is not None:
                write_arm_pose(pose, SERVO_MOVE_TIME_MS)

    # ---------- 主追踪 ----------
    def track(self, frame):
        frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # 降采样检测
        detect_frame = cv.resize(
            frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE,
            interpolation=cv.INTER_AREA
        )
        gray = cv.cvtColor(detect_frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        detect_min_size = (
            max(10, int(MIN_FACE_SIZE[0] * DETECTION_SCALE)),
            max(10, int(MIN_FACE_SIZE[1] * DETECTION_SCALE)),
        )
        small_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CASCADE_SCALE_FACTOR,
            minNeighbors=CASCADE_MIN_NEIGHBORS,
            minSize=detect_min_size,
        )

        inv_scale = 1.0 / DETECTION_SCALE
        faces = [
            tuple(int(round(v * inv_scale)) for v in face)
            for face in small_faces
        ]
        face = self.face_filter(faces)

        if face is None:
            self.reset_pid()
            cv.putText(frame, 'NO FACE', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.drawMarker(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
                         (255, 255, 255), cv.MARKER_CROSS, 24, 2)
            return frame, False

        x, y, w, h = face
        raw_center = (x + w // 2, y + h // 2)

        # EMA 平滑
        self.center_history.append(raw_center)
        median_center = (
            float(median(p[0] for p in self.center_history)),
            float(median(p[1] for p in self.center_history)),
        )
        if self.smoothed_center is None:
            self.smoothed_center = median_center
        else:
            self.smoothed_center = (
                PAN_SMOOTHING_ALPHA  * median_center[0]
                + (1.0 - PAN_SMOOTHING_ALPHA)  * self.smoothed_center[0],
                TILT_SMOOTHING_ALPHA * median_center[1]
                + (1.0 - TILT_SMOOTHING_ALPHA) * self.smoothed_center[1],
            )
        center_x, center_y = (
            int(round(self.smoothed_center[0])),
            int(round(self.smoothed_center[1])),
        )

        self._update_servos(center_x, center_y)

        # 绘标记
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv.drawMarker(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
                     (255, 255, 255), cv.MARKER_CROSS, 24, 2)
        cv.line(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
               (center_x, center_y), (255, 255, 0), 2)
        cv.putText(frame,
                   f'FACE ({center_x},{center_y}) '
                   f'S1:{self.pan_angle:.0f} S2:{self.tilt_angle:.0f} '
                   f'S3:{self.tilt_aux_angle:.0f}',
                   (10, FRAME_HEIGHT - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return frame, True
    
tracker = FaceTrack(Arm) if FACE_MODEL_PATH else None
model = 'General'

_fps_last_time   = time.time()
_fps_frame_count = 0
_fps_display     = 0.0

button_layout = widgets.Layout(width='200px', height='100px', align_self='center')
output = widgets.Output()

exit_button = widgets.Button(
    description='Exit', button_style='danger', layout=button_layout
)
imgbox = widgets.Image(
    format='jpg', height=480, width=640,
    layout=widgets.Layout(align_self='auto')
)
controls_box = widgets.VBox(
    [imgbox, exit_button],
    layout=widgets.Layout(align_self='auto')
)

def exit_button_Callback(value):
    global model
    model = 'Exit'

exit_button.on_click(exit_button_Callback)
display(controls_box, output)

def camera():
    global model, _fps_last_time, _fps_frame_count, _fps_display

    capture = cv.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        with output:
            print(f'[ERROR] 无法打开摄像头 /dev/video{CAMERA_INDEX}')
        return

    capture.set(3, FRAME_WIDTH)
    capture.set(4, FRAME_HEIGHT)
    capture.set(5, CAMERA_FPS)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

    with output:
        print(f'[OK] 摄像头已打开（/dev/video{CAMERA_INDEX}）')
        print('人脸追踪已启动...点击 Exit 退出')

    frame_count = 0
    while capture.isOpened() and model != 'Exit':
        try:
            ret, img = capture.read()
            if not ret:
                sleep(0.005)
                continue

            if FLIP_IMAGE:
                img = cv.flip(img, 1)

            if tracker is not None:
                img, has_face = tracker.track(img)
            else:
                cv.putText(img, 'Face model not loaded', (120, 240),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS
            _fps_frame_count += 1
            now = time.time()
            if now - _fps_last_time >= 1.0:
                _fps_display = _fps_frame_count / (now - _fps_last_time)
                _fps_frame_count = 0
                _fps_last_time = now
            cv.putText(img, f'FPS: {_fps_display:.1f}', (540, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 隔帧编码
            if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                ret, jpg = cv.imencode('.jpg', img,
                                       [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ret:
                    imgbox.value = jpg.tobytes()
            frame_count += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            with output:
                print(f'[WARN] {type(e).__name__}: {e}')
            sleep(0.1)

    capture.release()
    cv.destroyAllWindows()
    with output:
        print('[INFO] 程序已退出，摄像头资源已释放')


if CAMERA_INDEX is not None and tracker is not None:
    threading.Thread(target=camera, daemon=True).start()
    print('[INFO] 追踪线程已启动')
elif CAMERA_INDEX is None:
    print('[ERROR] 未探测到摄像头设备')
else:
    print('[ERROR] 人脸检测模型未加载')