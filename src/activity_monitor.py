"""
Activity Monitor
"""

import time
from collections import deque
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

from config import ACTIVITY_CONFIG
from pose_estimator import PoseResult


@dataclass
class ActivityEvent:
    timestamp: float
    event_type: str
    duration: float


class ActivityMonitor:
    def __init__(self, config=ACTIVITY_CONFIG):
        self.config = config
        self._position_history = deque(maxlen=config.history_size)
        self._movement_history = deque(maxlen=config.history_size)
        self._last_movement = time.time()
        self._alert_sent = False
        self._callbacks = []

    def process(self, pose: Optional[PoseResult], person_detected: bool) -> Dict:
        current = time.time()

        if not person_detected or pose is None:
            return {
                'is_active': False,
                'movement': 0.0,
                'inactive_duration': current - self._last_movement,
                'inactivity_alert': self._alert_sent,
                'person_detected': False
            }

        center = pose.get_center_of_mass()
        movement = 0.0

        if self._position_history:
            last = self._position_history[-1]
            movement = ((center[0]-last[0])**2 + (center[1]-last[1])**2)**0.5

        self._position_history.append(center)
        self._movement_history.append(movement)

        if movement > self.config.movement_threshold:
            self._last_movement = current
            if self._alert_sent:
                self._alert_sent = False
                print("Activity resumed")

        inactive_dur = current - self._last_movement

        if inactive_dur > self.config.inactivity_timeout and not self._alert_sent:
            self._alert_sent = True
            print(f"INACTIVITY: No movement for {inactive_dur/60:.1f} min")
            for cb in self._callbacks:
                try:
                    cb(ActivityEvent(current, "inactivity_alert", inactive_dur))
                except:
                    pass

        return {
            'is_active': movement > self.config.movement_threshold,
            'movement': movement,
            'inactive_duration': inactive_dur,
            'inactivity_alert': self._alert_sent,
            'person_detected': True
        }

    def register_inactivity_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def reset(self):
        self._position_history.clear()
        self._movement_history.clear()
        self._last_movement = time.time()
        self._alert_sent = False
