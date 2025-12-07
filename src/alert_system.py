"""
Alert System
"""

import time
import threading
from typing import Dict, List
from dataclasses import dataclass
from queue import Queue

from config import ALERT_CONFIG


@dataclass
class Alert:
    timestamp: float
    alert_type: str
    priority: str
    message: str


class AlertSystem:
    def __init__(self, config=ALERT_CONFIG):
        self.config = config
        self._queue = Queue()
        self._history = []
        self._last_alert = {}
        self._running = False
        self._worker = None

    def initialize(self) -> bool:
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        print("Alert system initialized")
        return True

    def _loop(self):
        while self._running:
            try:
                alert = self._queue.get(timeout=1.0)
            except:
                continue

            last = self._last_alert.get(alert.alert_type, 0)
            if time.time() - last < self.config.alert_cooldown:
                continue

            self._deliver(alert)
            self._history.append(alert)
            self._last_alert[alert.alert_type] = time.time()

    def _deliver(self, alert: Alert):
        icon = "🚨" if alert.priority == "high" else "⚠️"
        print(f"\n{icon} ALERT: {alert.message}")

    def send_fall_alert(self, event):
        msg = f"FALL in {self.config.location_name}! Type: {event.fall_type}, Conf: {event.confidence:.0%}"
        self._queue.put(Alert(event.timestamp, "fall", "high", msg))

    def send_inactivity_alert(self, event):
        msg = f"INACTIVITY in {self.config.location_name}! No movement for {event.duration/60:.1f} min"
        self._queue.put(Alert(event.timestamp, "inactivity", "medium", msg))

    def send_test_alert(self):
        self._queue.put(Alert(time.time(), "test", "low", f"Test from {self.config.location_name}"))

    def get_stats(self) -> Dict:
        return {
            'total': len(self._history),
            'falls': sum(1 for a in self._history if a.alert_type == "fall"),
            'inactivity': sum(1 for a in self._history if a.alert_type == "inactivity")
        }

    def shutdown(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=2.0)
        print("Alert system shutdown")
