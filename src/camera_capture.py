"""
Camera Capture Module
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple
from dataclasses import dataclass

from config import CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, MOVENET_INPUT_SIZE


@dataclass
class CameraInfo:
    device: str
    name: str
    width: int
    height: int
    fps: int


class CameraCapture:
    def __init__(self, device: str = CAMERA_DEVICE, width: int = CAMERA_WIDTH,
                 height: int = CAMERA_HEIGHT, fps: int = CAMERA_FPS):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None
        self._actual_fps = 0.0

    def initialize(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                print(f"Failed to open camera: {self.device}")
                return False

            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for _ in range(10):
                self._cap.read()

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            print(f"Camera initialized: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Camera init error: {e}")
            return False

    def _capture_loop(self):
        fps_counter = 0
        fps_timer = time.time()
        while self._running:
            if self._cap is None:
                time.sleep(0.01)
                continue
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self._actual_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_preprocessed_frame(self, size: int = MOVENET_INPUT_SIZE) -> Optional[np.ndarray]:
        frame = self.get_frame()
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (size, size))
        return np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    def get_fps(self) -> float:
        return self._actual_fps

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("Camera released")
