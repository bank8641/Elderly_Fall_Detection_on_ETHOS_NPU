#!/usr/bin/env python3
"""
Elderly Fall Detection - Direct Framebuffer Display
NXP i.MX93 + Arm Ethos-U65 NPU
No PyGame - writes directly to /dev/fb0
"""

import sys
import os
import time
import signal
import mmap
import struct

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import CAMERA_DEVICE, SystemState, ActivityState
from camera_capture import CameraCapture
from pose_estimator import PoseEstimator
from fall_detector import FallDetector
from activity_monitor import ActivityMonitor
from alert_system import AlertSystem


class FramebufferDisplay:
    """Direct framebuffer display - no window system overhead"""

    def __init__(self, device='/dev/fb0'):
        self.device = device
        self.fb = None
        self.fbmap = None
        self.width = 0
        self.height = 0
        self.bpp = 32
        self.line_length = 0

    def initialize(self):
        try:
            # Read framebuffer info
            with open('/sys/class/graphics/fb0/virtual_size', 'r') as f:
                size = f.read().strip().split(',')
                self.width = int(size[0])
                self.height = int(size[1])

            with open('/sys/class/graphics/fb0/bits_per_pixel', 'r') as f:
                self.bpp = int(f.read().strip())

            with open('/sys/class/graphics/fb0/stride', 'r') as f:
                self.line_length = int(f.read().strip())
        except:
            # Default to common values
            self.width = 1920
            self.height = 1080
            self.bpp = 32
            self.line_length = self.width * 4

        try:
            self.fb = open(self.device, 'r+b')
            fb_size = self.line_length * self.height
            self.fbmap = mmap.mmap(self.fb.fileno(), fb_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
            print(f"Framebuffer: {self.width}x{self.height} @ {self.bpp}bpp")
            return True
        except Exception as e:
            print(f"Framebuffer error: {e}")
            return False

    def write_frame(self, frame):
        """Write BGR frame to framebuffer"""
        if self.fbmap is None:
            return

        # Resize to screen
        frame_resized = cv2.resize(frame, (self.width, self.height))

        # Convert BGR to BGRA (32-bit)
        if self.bpp == 32:
            frame_bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
            self.fbmap.seek(0)
            self.fbmap.write(frame_bgra.tobytes())
        elif self.bpp == 16:
            # Convert to RGB565
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            r = (frame_rgb[:,:,0] >> 3).astype(np.uint16)
            g = (frame_rgb[:,:,1] >> 2).astype(np.uint16)
            b = (frame_rgb[:,:,2] >> 3).astype(np.uint16)
            rgb565 = (r << 11) | (g << 5) | b
            self.fbmap.seek(0)
            self.fbmap.write(rgb565.tobytes())

    def cleanup(self):
        if self.fbmap:
            self.fbmap.close()
        if self.fb:
            self.fb.close()


class FallDetectionApp:
    def __init__(self, camera_device=CAMERA_DEVICE):
        self.camera_device = camera_device
        self.camera = None
        self.pose_estimator = None
        self.fall_detector = None
        self.activity_monitor = None
        self.alert_system = None
        self.display = None

        self.state = SystemState.INITIALIZING
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0

        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, signum, frame):
        print("\nShutdown...")
        self.running = False

    def initialize(self):
        print("=" * 40)
        print("Fall Detection - Framebuffer Mode")
        print("=" * 40)

        print("\n[1/5] Framebuffer...")
        self.display = FramebufferDisplay()
        if not self.display.initialize():
            print("Framebuffer failed, continuing without display")
            self.display = None

        print("\n[2/5] Camera...")
        self.camera = CameraCapture(device=self.camera_device)
        if not self.camera.initialize():
            return False

        print("\n[3/5] Pose Estimator...")
        self.pose_estimator = PoseEstimator()
        if not self.pose_estimator.initialize():
            return False
        print(f"  NPU: {'ON' if self.pose_estimator.is_using_npu() else 'OFF'}")

        print("\n[4/5] Fall Detector...")
        self.fall_detector = FallDetector()
        self.fall_detector.register_fall_callback(self._on_fall)

        print("\n[5/5] Alert System...")
        self.activity_monitor = ActivityMonitor()
        self.alert_system = AlertSystem()
        self.alert_system.initialize()

        self.state = SystemState.RUNNING
        print("\nReady! Ctrl+C to quit")
        return True

    def _on_fall(self, event):
        self.state = SystemState.FALL_DETECTED
        self.alert_system.send_fall_alert(event)
        print(f">>> FALL DETECTED! {event.fall_type} conf={event.confidence:.0%}")

    def run(self):
        self.running = True
        self.start_time = time.time()
        last_fps_time = self.start_time
        fps_count = 0

        det_state = None
        pose = None

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            # Pose estimation
            preprocessed = self.camera.get_preprocessed_frame()
            if preprocessed is not None:
                pose = self.pose_estimator.estimate(preprocessed)

            # Fall detection
            if pose:
                det_state = self.fall_detector.process(pose)
                if det_state.is_fall_detected:
                    self.state = SystemState.FALL_DETECTED
                elif self.state == SystemState.FALL_DETECTED:
                    if det_state.consecutive_standing_frames > 10:
                        self.state = SystemState.RUNNING

                person = len(pose.get_valid_keypoints(0.3)) >= 5
                self.activity_monitor.process(pose, person)

            # Render and display
            display_frame = self._render(frame, pose, det_state)
            if self.display:
                self.display.write_frame(display_frame)

            # FPS
            self.frame_count += 1
            fps_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                self.fps = fps_count
                fps_count = 0
                last_fps_time = now
                print(f"FPS: {self.fps:.0f} | State: {self.state.value}", end='\r')

        self.cleanup()

    def _render(self, frame, pose, det_state):
        """Render frame with overlays"""
        h, w = frame.shape[:2]

        # Draw skeleton
        if pose:
            frame = self._draw_skeleton(frame, pose)

        # Status bar at top
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # NPU status
        npu = "NPU:ON" if self.pose_estimator.is_using_npu() else "CPU"
        cv2.putText(frame, npu, (150, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Inference time
        if pose:
            cv2.putText(frame, f"{pose.inference_time_ms:.0f}ms", (260, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Activity and angle
        if det_state:
            cv2.putText(frame, f"{det_state.activity_state.value} | {det_state.body_angle:.0f}deg",
                       (350, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Fall alert
        if self.state == SystemState.FALL_DETECTED:
            # Red border
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)

            # Alert banner
            cv2.rectangle(frame, (0, h//2-40), (w, h//2+40), (0, 0, 200), -1)
            cv2.putText(frame, "FALL DETECTED!", (w//2-150, h//2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return frame

    def _draw_skeleton(self, frame, pose):
        """Draw pose skeleton on frame"""
        h, w = frame.shape[:2]
        kps = pose.keypoints
        conf = 0.3

        connections = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)
        ]

        for i, j in connections:
            if kps[i].confidence >= conf and kps[j].confidence >= conf:
                x1, y1 = int(kps[i].x * w), int(kps[i].y * h)
                x2, y2 = int(kps[j].x * w), int(kps[j].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for kp in kps:
            if kp.confidence >= conf:
                x, y = int(kp.x * w), int(kp.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        return frame

    def cleanup(self):
        print("\nCleaning up...")
        if self.display:
            self.display.cleanup()
        if self.camera:
            self.camera.release()
        if self.alert_system:
            self.alert_system.shutdown()

        rt = time.time() - self.start_time
        if self.frame_count > 0:
            print(f"{self.frame_count} frames in {rt:.1f}s = {self.frame_count/rt:.1f} FPS")


def main():
    app = FallDetectionApp()
    if app.initialize():
        app.run()

if __name__ == "__main__":
    main()
