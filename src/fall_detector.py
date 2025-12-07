"""
Fall Detection Algorithm - FIXED VERSION
More robust angle calculation with fallbacks
"""

import math
import time
from collections import deque
from typing import Optional, List, Callable
from dataclasses import dataclass

from config import FALL_CONFIG, Keypoint, UPPER_BODY_KEYPOINTS, LOWER_BODY_KEYPOINTS, ActivityState
from pose_estimator import PoseResult


@dataclass
class FallEvent:
    timestamp: float
    confidence: float
    fall_type: str
    body_angle: float
    velocity: float
    duration_frames: int


@dataclass
class DetectionState:
    is_fall_detected: bool = False
    fall_confidence: float = 0.0
    consecutive_fall_frames: int = 0
    consecutive_standing_frames: int = 0
    activity_state: ActivityState = ActivityState.UNKNOWN
    body_angle: float = 0.0
    vertical_velocity: float = 0.0


class FallDetector:
    def __init__(self, config=FALL_CONFIG):
        self.config = config
        self.state = DetectionState()
        self._pose_history = deque(maxlen=30)
        self._center_history = deque(maxlen=30)
        self._time_history = deque(maxlen=30)
        self._fall_events = []
        self._fall_callbacks = []

    def process(self, pose: PoseResult) -> DetectionState:
        current_time = time.time()
        self._pose_history.append(pose)
        self._time_history.append(current_time)

        valid = pose.get_valid_keypoints(self.config.confidence_threshold)
        if len(valid) < 5:
            self.state.activity_state = ActivityState.NOT_DETECTED
            return self.state

        body_angle = self._calc_body_angle(pose)
        velocity = self._calc_velocity()
        y_diff = self._calc_y_diff(pose)
        bbox_ratio = self._calc_bbox_ratio(pose)
        x_spread = self._calc_x_spread(pose)

        center = pose.get_center_of_mass()
        self._center_history.append(center)

        self.state.body_angle = body_angle
        self.state.vertical_velocity = velocity
        self.state.activity_state = self._classify_activity(body_angle, bbox_ratio, y_diff)

        is_falling = self._check_fall(body_angle, velocity, y_diff, bbox_ratio, x_spread)

        if is_falling:
            self.state.consecutive_fall_frames += 1
            self.state.consecutive_standing_frames = 0
        else:
            self.state.consecutive_standing_frames += 1
            if self.state.consecutive_standing_frames >= self.config.recovery_frames:
                self.state.consecutive_fall_frames = 0

        if self.state.consecutive_fall_frames >= self.config.confirm_frames and not self.state.is_fall_detected:
            self._trigger_fall(pose, body_angle, velocity)

        if not is_falling and self.state.is_fall_detected:
            if self.state.consecutive_standing_frames >= self.config.recovery_frames:
                self.state.is_fall_detected = False
                self.state.fall_confidence = 0.0
                print("Recovery detected")

        return self.state

    def _calc_body_angle(self, pose: PoseResult) -> float:
        conf = self.config.confidence_threshold

        ls = pose.get_keypoint(Keypoint.LEFT_SHOULDER)
        rs = pose.get_keypoint(Keypoint.RIGHT_SHOULDER)
        lh = pose.get_keypoint(Keypoint.LEFT_HIP)
        rh = pose.get_keypoint(Keypoint.RIGHT_HIP)
        nose = pose.get_keypoint(Keypoint.NOSE)

        shoulders_ok = ls and rs and ls.confidence >= conf and rs.confidence >= conf
        hips_ok = lh and rh and lh.confidence >= conf and rh.confidence >= conf

        if shoulders_ok and hips_ok:
            sx = (ls.x + rs.x) / 2
            sy = (ls.y + rs.y) / 2
            hx = (lh.x + rh.x) / 2
            hy = (lh.y + rh.y) / 2
            return self._angle_from_points(sx, sy, hx, hy)

        shoulder = None
        if ls and ls.confidence >= conf:
            shoulder = ls
        elif rs and rs.confidence >= conf:
            shoulder = rs

        hip = None
        if lh and lh.confidence >= conf:
            hip = lh
        elif rh and rh.confidence >= conf:
            hip = rh

        if shoulder and hip:
            return self._angle_from_points(shoulder.x, shoulder.y, hip.x, hip.y)

        if nose and nose.confidence >= conf and hip:
            return self._angle_from_points(nose.x, nose.y, hip.x, hip.y)

        bbox_ratio = self._calc_bbox_ratio(pose)
        if bbox_ratio < 0.6:
            return 75.0
        elif bbox_ratio < 1.0:
            return 45.0
        elif bbox_ratio < 1.5:
            return 20.0
        else:
            return 5.0

    def _angle_from_points(self, x1, y1, x2, y2) -> float:
        dx = x1 - x2
        dy = y1 - y2
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        return max(0, min(90, angle))

    def _calc_velocity(self) -> float:
        if len(self._center_history) < 2:
            return 0.0
        centers = list(self._center_history)[-5:]
        times = list(self._time_history)[-5:]
        if len(centers) < 2:
            return 0.0
        total_dy = sum(centers[i][1] - centers[i-1][1] for i in range(1, len(centers)))
        total_dt = times[-1] - times[0]
        return total_dy / total_dt if total_dt > 0 else 0.0

    def _calc_y_diff(self, pose: PoseResult) -> float:
        conf = self.config.confidence_threshold
        upper = [pose.get_keypoint(k) for k in UPPER_BODY_KEYPOINTS]
        lower = [pose.get_keypoint(k) for k in LOWER_BODY_KEYPOINTS]
        upper_y = [k.y for k in upper if k and k.confidence >= conf]
        lower_y = [k.y for k in lower if k and k.confidence >= conf]
        if not upper_y or not lower_y:
            return 1.0
        return abs(sum(lower_y)/len(lower_y) - sum(upper_y)/len(upper_y))

    def _calc_bbox_ratio(self, pose: PoseResult) -> float:
        x1, y1, x2, y2 = pose.get_bounding_box()
        w = x2 - x1
        h = y2 - y1
        return h / w if w > 0.01 else 2.0

    def _calc_x_spread(self, pose: PoseResult) -> float:
        valid = pose.get_valid_keypoints(self.config.confidence_threshold)
        if len(valid) < 2:
            return 0.0
        xs = [k.x for k in valid]
        return max(xs) - min(xs)

    def _classify_activity(self, angle: float, bbox: float, y_diff: float) -> ActivityState:
        if angle > 60 or y_diff < 0.1 or bbox < 0.6:
            return ActivityState.LYING_DOWN
        if 30 < angle < 60 and bbox < 1.2:
            return ActivityState.SITTING
        if angle < 30 and bbox > 1.0:
            return ActivityState.STANDING
        return ActivityState.UNKNOWN

    def _check_fall(self, angle, vel, y_diff, bbox, spread) -> bool:
        c = self.config
        a = angle > c.angle_threshold
        v = vel > c.velocity_threshold
        y = y_diff < c.y_diff_threshold
        b = bbox < c.height_ratio_threshold
        s = spread > c.x_spread_threshold
        count = sum([a, v, y, b, s])
        return (a and v) or (a and y) or (y and b) or count >= 3

    def _trigger_fall(self, pose, angle, vel):
        self.state.is_fall_detected = True
        self.state.activity_state = ActivityState.FALLEN
        conf = min(1.0, 0.3*(angle/90) + 0.25*(vel/0.3) + 0.25 + 0.2)
        self.state.fall_confidence = conf
        fall_type = "rapid_fall" if vel > 0.3 else "collapse" if angle > 70 else "gradual_fall"
        event = FallEvent(time.time(), conf, fall_type, angle, vel, self.state.consecutive_fall_frames)
        self._fall_events.append(event)
        print(f"FALL DETECTED! Type: {fall_type}, Confidence: {conf:.0%}, Angle: {angle:.1f}")
        for cb in self._fall_callbacks:
            try:
                cb(event)
            except:
                pass

    def register_fall_callback(self, cb: Callable):
        self._fall_callbacks.append(cb)

    def get_fall_events(self) -> List[FallEvent]:
        return self._fall_events.copy()

    def reset(self):
        self.state = DetectionState()
        self._pose_history.clear()
        self._center_history.clear()
        self._time_history.clear()
