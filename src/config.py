"""
Configuration for Elderly Fall Detection System
"""

import os
from dataclasses import dataclass
from enum import Enum

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Camera
CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# NPU
NPU_DELEGATE_PATH = "/usr/lib/libethosu_delegate.so"
USE_NPU = True

# Model
MOVENET_MODEL_PATH = os.path.join(MODEL_DIR, "movenet_lightning_int8_vela.tflite")
MOVENET_INPUT_SIZE = 192


class Keypoint(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_EDGES = [
    (Keypoint.NOSE, Keypoint.LEFT_EYE),
    (Keypoint.NOSE, Keypoint.RIGHT_EYE),
    (Keypoint.LEFT_EYE, Keypoint.LEFT_EAR),
    (Keypoint.RIGHT_EYE, Keypoint.RIGHT_EAR),
    (Keypoint.NOSE, Keypoint.LEFT_SHOULDER),
    (Keypoint.NOSE, Keypoint.RIGHT_SHOULDER),
    (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW),
    (Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST),
    (Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE),
    (Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE),
    (Keypoint.LEFT_KNEE, Keypoint.LEFT_ANKLE),
    (Keypoint.RIGHT_KNEE, Keypoint.RIGHT_ANKLE),
]

UPPER_BODY_KEYPOINTS = [Keypoint.NOSE, Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER]
LOWER_BODY_KEYPOINTS = [Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP, Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE]


@dataclass
class FallDetectionConfig:
    confidence_threshold: float = 0.3
    angle_threshold: float = 50.0
    velocity_threshold: float = 0.15
    confirm_frames: int = 3
    y_diff_threshold: float = 0.1
    x_spread_threshold: float = 0.4
    height_ratio_threshold: float = 0.8
    recovery_frames: int = 10


@dataclass 
class ActivityConfig:
    inactivity_timeout: float = 300.0
    movement_threshold: float = 0.02
    history_size: int = 300


@dataclass
class AlertConfig:
    enable_gpio_alarm: bool = False
    enable_mqtt_alert: bool = False
    gpio_buzzer_pin: int = 17
    gpio_led_pin: int = 27
    buzzer_duration: float = 5.0
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "elderly/fall_alert"
    alert_cooldown: float = 30.0
    location_name: str = "Living Room"


FALL_CONFIG = FallDetectionConfig()
ACTIVITY_CONFIG = ActivityConfig()
ALERT_CONFIG = AlertConfig()


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    FALL_DETECTED = "fall_detected"
    ALERT_SENT = "alert_sent"
    INACTIVE = "inactive"
    ERROR = "error"


class ActivityState(Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    LYING_DOWN = "lying_down"
    WALKING = "walking"
    FALLEN = "fallen"
    NOT_DETECTED = "not_detected"
