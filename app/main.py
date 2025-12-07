#!/usr/bin/env python3
"""
Elderly Fall Detection System - Main Application
NXP i.MX93 + Arm Ethos-U65 NPU

Usage:
    python3 main.py [--no-display] [--camera /dev/videoX]

Controls:
    q - Quit
    t - Test alert
    s - Statistics
    r - Reset
"""

import sys
import os
import time
import signal
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import CAMERA_DEVICE, SystemState, ActivityState
from camera_capture import CameraCapture
from pose_estimator import PoseEstimator, visualize_pose
from fall_detector import FallDetector
from activity_monitor import ActivityMonitor
from alert_system import AlertSystem


class FallDetectionApp:
    def __init__(self, camera_device=CAMERA_DEVICE, enable_display=True):
        self.camera_device = camera_device
        self.enable_display = enable_display

        self.camera = None
        self.pose_estimator = None
        self.fall_detector = None
        self.activity_monitor = None
        self.alert_system = None

        self.state = SystemState.INITIALIZING
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0

        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, signum, frame):
        print(f"\nShutdown signal received...")
        self.running = False

    def initialize(self) -> bool:
        print("=" * 50)
        print("Elderly Fall Detection System")
        print("NXP i.MX93 + Ethos-U65 NPU")
        print("=" * 50)

        print("\n[1/5] Camera...")
        self.camera = CameraCapture(device=self.camera_device)
        if not self.camera.initialize():
            return False

        print("\n[2/5] Pose Estimator...")
        self.pose_estimator = PoseEstimator()
        if not self.pose_estimator.initialize():
            return False
        print(f"  NPU: {'ON' if self.pose_estimator.is_using_npu() else 'OFF'}")

        print("\n[3/5] Fall Detector...")
        self.fall_detector = FallDetector()
        self.fall_detector.register_fall_callback(self._on_fall)

        print("\n[4/5] Activity Monitor...")
        self.activity_monitor = ActivityMonitor()
        self.activity_monitor.register_inactivity_callback(self._on_inactivity)

        print("\n[5/5] Alert System...")
        self.alert_system = AlertSystem()
        self.alert_system.initialize()

        self.state = SystemState.RUNNING
        print("\n" + "=" * 50)
        print("System Ready!")
        print("Controls: q=quit, t=test, s=stats, r=reset")
        print("=" * 50 + "\n")
        return True

    def _on_fall(self, event):
        self.state = SystemState.FALL_DETECTED
        self.alert_system.send_fall_alert(event)

    def _on_inactivity(self, event):
        self.state = SystemState.INACTIVE
        self.alert_system.send_inactivity_alert(event)

    def run(self):
        self.running = True
        self.start_time = time.time()
        last_fps_time = self.start_time

        print("Running... Press 'q' to quit\n")

        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                preprocessed = self.camera.get_preprocessed_frame()
                pose_result = self.pose_estimator.estimate(preprocessed) if preprocessed is not None else None

                detection_state = None
                if pose_result:
                    detection_state = self.fall_detector.process(pose_result)
                    if detection_state.is_fall_detected:
                        self.state = SystemState.FALL_DETECTED
                    elif self.state == SystemState.FALL_DETECTED and not detection_state.is_fall_detected:
                        self.state = SystemState.RUNNING

                person_detected = pose_result and len(pose_result.get_valid_keypoints(0.3)) >= 5
                activity = self.activity_monitor.process(pose_result, person_detected)

                if self.enable_display:
                    display = self._render(frame, pose_result, detection_state, activity)
                    cv2.imshow('Fall Detection', display)
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key):
                        break

                self.frame_count += 1
                if time.time() - last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.start_time)
                    last_fps_time = time.time()

        finally:
            self.cleanup()

    def _render(self, frame, pose, det_state, activity):
        display = frame.copy()
        h, w = display.shape[:2]

        if pose:
            display = visualize_pose(display, pose)

        # Top bar
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(display, "FALL DETECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_txt = f"FPS: {self.fps:.1f}"
        if pose:
            fps_txt += f" | Inf: {pose.inference_time_ms:.0f}ms"
        cv2.putText(display, fps_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        npu = "NPU:ON" if self.pose_estimator.is_using_npu() else "CPU"
        color = (0, 255, 0) if self.pose_estimator.is_using_npu() else (0, 165, 255)
        cv2.putText(display, npu, (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if det_state:
            act = det_state.activity_state.value.upper()
            cv2.putText(display, f"Activity: {act} | Angle: {det_state.body_angle:.0f}",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Fall alert
        if det_state and det_state.is_fall_detected:
            cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 0, 255), 8)
            cv2.rectangle(display, (0, h//2-35), (w, h//2+35), (0, 0, 200), -1)
            cv2.putText(display, "FALL DETECTED!", (w//2-120, h//2+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Inactivity
        elif activity and activity.get('inactivity_alert'):
            cv2.rectangle(display, (0, h-40), (w, h), (0, 165, 255), -1)
            mins = activity.get('inactive_duration', 0) / 60
            cv2.putText(display, f"INACTIVITY: {mins:.1f} min", (w//2-100, h-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display

    def _handle_key(self, key) -> bool:
        if key == ord('q') or key == 27:
            return False
        elif key == ord('t'):
            self.alert_system.send_test_alert()
        elif key == ord('s'):
            self._print_stats()
        elif key == ord('r'):
            self.fall_detector.reset()
            self.activity_monitor.reset()
            self.state = SystemState.RUNNING
            print("State reset")
        return True

    def _print_stats(self):
        print("\n" + "=" * 40)
        print("STATISTICS")
        print("=" * 40)
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime/60:.1f} min")
        print(f"Frames: {self.frame_count}")
        print(f"FPS: {self.fps:.1f}")
        print(f"Avg inference: {self.pose_estimator.get_average_inference_time():.1f}ms")
        print(f"Falls detected: {len(self.fall_detector.get_fall_events())}")
        print(f"Alerts: {self.alert_system.get_stats()}")
        print("=" * 40 + "\n")

    def cleanup(self):
        print("\nCleaning up...")
        if self.enable_display:
            cv2.destroyAllWindows()
        if self.camera:
            self.camera.release()
        if self.alert_system:
            self.alert_system.shutdown()

        if self.frame_count > 0:
            runtime = time.time() - self.start_time
            print(f"Processed {self.frame_count} frames in {runtime:.1f}s ({self.frame_count/runtime:.1f} FPS)")
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Elderly Fall Detection')
    parser.add_argument('--camera', '-c', default=CAMERA_DEVICE, help='Camera device')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    args = parser.parse_args()

    app = FallDetectionApp(
        camera_device=args.camera,
        enable_display=not args.no_display
    )

    if app.initialize():
        app.run()
    else:
        print("Initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
