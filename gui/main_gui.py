#!/usr/bin/env python3
"""
Elderly Fall Detection System - GUI Application
NXP i.MX93 + Arm Ethos-U65 NPU

Displays on HDMI via Weston/Wayland
"""

import sys
import os
import time
import signal

# Force OpenCV to use GTK or Wayland backend
os.environ['QT_QPA_PLATFORM'] = 'wayland'

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import CAMERA_DEVICE, SystemState, ActivityState
from camera_capture import CameraCapture
from pose_estimator import PoseEstimator, visualize_pose
from fall_detector import FallDetector
from activity_monitor import ActivityMonitor
from alert_system import AlertSystem


class FallDetectionGUI:
    """Full-screen GUI for fall detection on HDMI display"""

    def __init__(self, camera_device=CAMERA_DEVICE):
        self.camera_device = camera_device

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

        # Colors (BGR)
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_ORANGE = (0, 165, 255)
        self.COLOR_CYAN = (255, 255, 0)

        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, signum, frame):
        print("\nShutdown...")
        self.running = False

    def initialize(self) -> bool:
        print("Initializing Fall Detection System...")

        print("[1/5] Camera...")
        self.camera = CameraCapture(device=self.camera_device)
        if not self.camera.initialize():
            print("Camera failed!")
            return False

        print("[2/5] Pose Estimator...")
        self.pose_estimator = PoseEstimator()
        if not self.pose_estimator.initialize():
            print("Pose estimator failed!")
            return False

        print("[3/5] Fall Detector...")
        self.fall_detector = FallDetector()
        self.fall_detector.register_fall_callback(self._on_fall)

        print("[4/5] Activity Monitor...")
        self.activity_monitor = ActivityMonitor()
        self.activity_monitor.register_inactivity_callback(self._on_inactivity)

        print("[5/5] Alert System...")
        self.alert_system = AlertSystem()
        self.alert_system.initialize()

        self.state = SystemState.RUNNING
        print("System Ready!")
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
        fps_frame_count = 0

        # Create fullscreen window
        window_name = 'Fall Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Running... Press 'q' to quit, 't' to test alert")

        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Pose estimation
                preprocessed = self.camera.get_preprocessed_frame()
                pose_result = None
                if preprocessed is not None:
                    pose_result = self.pose_estimator.estimate(preprocessed)

                # Fall detection
                detection_state = None
                if pose_result:
                    detection_state = self.fall_detector.process(pose_result)
                    if detection_state.is_fall_detected:
                        self.state = SystemState.FALL_DETECTED
                    elif self.state == SystemState.FALL_DETECTED and not detection_state.is_fall_detected:
                        self.state = SystemState.RUNNING

                # Activity monitoring
                person_detected = pose_result and len(pose_result.get_valid_keypoints(0.3)) >= 5
                activity = self.activity_monitor.process(pose_result, person_detected)

                # Render display
                display = self._render(frame, pose_result, detection_state, activity)
                cv2.imshow(window_name, display)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('t'):
                    self.alert_system.send_test_alert()
                elif key == ord('r'):
                    self.fall_detector.reset()
                    self.activity_monitor.reset()
                    self.state = SystemState.RUNNING

                # Update FPS
                self.frame_count += 1
                fps_frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    self.fps = fps_frame_count / (now - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = now

        finally:
            self.cleanup()

    def _render(self, frame, pose, det_state, activity):
        """Render the full display with all overlays"""
        # Get frame dimensions
        h, w = frame.shape[:2]

        # Create display canvas (larger than camera for UI elements)
        display_h = max(720, h)
        display_w = max(1280, w + 320)  # Extra space for side panel
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        display[:] = self.COLOR_BG

        # Calculate video position (centered in main area)
        video_w = display_w - 320
        video_h = display_h - 80
        scale = min(video_w / w, video_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        x_offset = (video_w - new_w) // 2
        y_offset = 70 + (video_h - new_h) // 2

        # Resize and place video frame
        video_frame = cv2.resize(frame, (new_w, new_h))

        # Draw pose on video
        if pose:
            video_frame = visualize_pose(video_frame, pose)

        display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = video_frame

        # Draw header bar
        cv2.rectangle(display, (0, 0), (display_w, 60), (40, 40, 40), -1)
        cv2.putText(display, "FALL DETECTION SYSTEM", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_TEXT, 2)

        # NPU indicator
        npu_active = self.pose_estimator.is_using_npu() if self.pose_estimator else False
        npu_text = "NPU: ON" if npu_active else "NPU: OFF"
        npu_color = self.COLOR_GREEN if npu_active else self.COLOR_ORANGE
        cv2.putText(display, npu_text, (video_w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, npu_color, 2)

        # Draw side panel
        panel_x = display_w - 300
        cv2.rectangle(display, (panel_x, 60), (display_w, display_h), (45, 45, 45), -1)
        cv2.line(display, (panel_x, 60), (panel_x, display_h), (60, 60, 60), 2)

        # Panel: Status section
        py = 100
        cv2.putText(display, "STATUS", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        py += 35

        # State indicator
        state_text = self.state.value.upper().replace('_', ' ')
        if self.state == SystemState.FALL_DETECTED:
            state_color = self.COLOR_RED
        elif self.state == SystemState.RUNNING:
            state_color = self.COLOR_GREEN
        else:
            state_color = self.COLOR_ORANGE
        cv2.putText(display, state_text, (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        py += 50

        # Panel: Metrics section
        cv2.putText(display, "METRICS", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        py += 35

        cv2.putText(display, f"FPS: {self.fps:.1f}", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_CYAN, 2)
        py += 30

        if pose:
            cv2.putText(display, f"Inference: {pose.inference_time_ms:.0f}ms", (panel_x + 20, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_CYAN, 2)
        py += 50

        # Panel: Detection section
        cv2.putText(display, "DETECTION", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        py += 35

        if det_state:
            activity_text = det_state.activity_state.value.replace('_', ' ').title()
            cv2.putText(display, f"Activity: {activity_text}", (panel_x + 20, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
            py += 28

            cv2.putText(display, f"Body Angle: {det_state.body_angle:.1f}", (panel_x + 20, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
            py += 28

            keypoints = len(pose.get_valid_keypoints(0.3)) if pose else 0
            cv2.putText(display, f"Keypoints: {keypoints}/17", (panel_x + 20, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
            py += 28

            if det_state.is_fall_detected:
                cv2.putText(display, f"Confidence: {det_state.fall_confidence:.0%}", (panel_x + 20, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_RED, 2)
        py += 50

        # Panel: Controls section
        cv2.putText(display, "CONTROLS", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        py += 30
        cv2.putText(display, "Q - Quit", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        py += 22
        cv2.putText(display, "T - Test Alert", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        py += 22
        cv2.putText(display, "R - Reset", (panel_x + 20, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # Fall alert overlay
        if det_state and det_state.is_fall_detected:
            # Red border around video
            cv2.rectangle(display, (x_offset-4, y_offset-4),
                         (x_offset+new_w+4, y_offset+new_h+4), self.COLOR_RED, 8)

            # Alert banner
            banner_h = 100
            banner_y = y_offset + (new_h - banner_h) // 2
            overlay = display.copy()
            cv2.rectangle(overlay, (x_offset, banner_y),
                         (x_offset + new_w, banner_y + banner_h), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            cv2.putText(display, "FALL DETECTED!",
                       (x_offset + new_w//2 - 180, banner_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, self.COLOR_TEXT, 3)

            cv2.putText(display, f"Confidence: {det_state.fall_confidence:.0%}",
                       (x_offset + new_w//2 - 100, banner_y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

        # Inactivity warning
        elif activity and activity.get('inactivity_alert'):
            mins = activity.get('inactive_duration', 0) / 60
            cv2.rectangle(display, (x_offset, y_offset + new_h - 50),
                         (x_offset + new_w, y_offset + new_h), self.COLOR_ORANGE, -1)
            cv2.putText(display, f"INACTIVITY WARNING: {mins:.1f} min",
                       (x_offset + 20, y_offset + new_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)

        # Timestamp footer
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, timestamp, (20, display_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        uptime = time.time() - self.start_time
        uptime_str = f"Uptime: {int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        cv2.putText(display, uptime_str, (video_w - 200, display_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return display

    def cleanup(self):
        print("Cleaning up...")
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.release()
        if self.alert_system:
            self.alert_system.shutdown()

        if self.frame_count > 0:
            runtime = time.time() - self.start_time
            print(f"Processed {self.frame_count} frames in {runtime:.1f}s ({self.frame_count/runtime:.1f} FPS)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fall Detection GUI')
    parser.add_argument('--camera', '-c', default=CAMERA_DEVICE, help='Camera device')
    args = parser.parse_args()

    app = FallDetectionGUI(camera_device=args.camera)

    if app.initialize():
        app.run()
    else:
        print("Initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
