#!/usr/bin/env python3
"""
Elderly Fall Detection - Ultra Optimized Display
NXP i.MX93 + Arm Ethos-U65 NPU
"""

import sys
import os
import time
import signal

os.environ.setdefault('SDL_VIDEODRIVER', 'wayland')

import pygame
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import CAMERA_DEVICE, SystemState, ActivityState
from camera_capture import CameraCapture
from pose_estimator import PoseEstimator
from fall_detector import FallDetector
from activity_monitor import ActivityMonitor
from alert_system import AlertSystem


class FallDetectionDisplay:
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

        self.screen = None
        self.screen_width = 800
        self.screen_height = 480

        # Only update display every N frames
        self.display_interval = 3

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.CYAN = (0, 255, 255)

        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, signum, frame):
        self.running = False

    def initialize(self):
        print("=" * 40)
        print("Fall Detection - Ultra Light")
        print("=" * 40)

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.FULLSCREEN
        )
        pygame.mouse.set_visible(False)
        self.font = pygame.font.Font(None, 48)
        self.font_big = pygame.font.Font(None, 72)

        print("[1/4] Camera...")
        self.camera = CameraCapture(device=self.camera_device)
        if not self.camera.initialize():
            return False

        print("[2/4] Pose Estimator...")
        self.pose_estimator = PoseEstimator()
        if not self.pose_estimator.initialize():
            return False
        print(f"  NPU: {'ON' if self.pose_estimator.is_using_npu() else 'OFF'}")

        print("[3/4] Fall Detector...")
        self.fall_detector = FallDetector()
        self.fall_detector.register_fall_callback(self._on_fall)

        print("[4/4] Activity Monitor...")
        self.activity_monitor = ActivityMonitor()
        self.alert_system = AlertSystem()
        self.alert_system.initialize()

        self.state = SystemState.RUNNING
        print("Ready! ESC to quit")
        return True

    def _on_fall(self, event):
        self.state = SystemState.FALL_DETECTED
        self.alert_system.send_fall_alert(event)
        print(f">>> FALL! {event.fall_type} conf={event.confidence:.0%}")

    def run(self):
        self.running = True
        self.start_time = time.time()
        last_fps = time.time()
        fps_count = 0
        frame_num = 0

        det_state = None
        pose = None

        # Pre-calculate sizes
        vid_w, vid_h = 560, 420
        vid_x, vid_y = 20, 40

        while self.running:
            # Events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running = False
                    elif ev.key == pygame.K_r:
                        self.fall_detector.reset()
                        self.state = SystemState.RUNNING

            # Get frame
            frame = self.camera.get_frame()
            if frame is None:
                continue

            # Process every frame for detection
            preprocessed = self.camera.get_preprocessed_frame()
            if preprocessed is not None:
                pose = self.pose_estimator.estimate(preprocessed)

            if pose:
                det_state = self.fall_detector.process(pose)
                if det_state.is_fall_detected:
                    self.state = SystemState.FALL_DETECTED
                elif self.state == SystemState.FALL_DETECTED:
                    if det_state.consecutive_standing_frames > 10:
                        self.state = SystemState.RUNNING

                person = len(pose.get_valid_keypoints(0.3)) >= 5
                self.activity_monitor.process(pose, person)

            # Update display less frequently
            frame_num += 1
            if frame_num % self.display_interval == 0:
                self._render_minimal(frame, pose, det_state, vid_x, vid_y, vid_w, vid_h)
                pygame.display.flip()

            # FPS
            self.frame_count += 1
            fps_count += 1
            now = time.time()
            if now - last_fps >= 1.0:
                self.fps = fps_count
                fps_count = 0
                last_fps = now

        self.cleanup()

    def _render_minimal(self, frame, pose, det_state, vx, vy, vw, vh):
        """Minimal rendering for speed"""
        self.screen.fill(self.BLACK)

        # Resize frame small and fast
        frame_small = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_NEAREST)

        # Draw simple skeleton directly on frame
        if pose:
            self._draw_skeleton_fast(frame_small, pose, vw, vh)

        # Convert to pygame
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        self.screen.blit(surf, (vx, vy))

        # Status panel on right
        px = 600
        py = 50

        # FPS
        self.screen.blit(self.font.render(f"FPS: {self.fps}", True, self.CYAN), (px, py))
        py += 40

        # Inference
        if pose:
            self.screen.blit(self.font.render(f"Inf: {pose.inference_time_ms:.0f}ms", True, self.CYAN), (px, py))
        py += 40

        # Angle
        if det_state:
            self.screen.blit(self.font.render(f"Angle: {det_state.body_angle:.0f}", True, self.WHITE), (px, py))
            py += 40
            self.screen.blit(self.font.render(f"{det_state.activity_state.value}", True, self.WHITE), (px, py))
        py += 60

        # Status
        if self.state == SystemState.FALL_DETECTED:
            # Big red alert
            pygame.draw.rect(self.screen, self.RED, (0, 0, self.screen_width, self.screen_height), 10)
            txt = self.font_big.render("FALL DETECTED!", True, self.RED)
            self.screen.blit(txt, (px - 50, py))
        else:
            txt = self.font.render("RUNNING", True, self.GREEN)
            self.screen.blit(txt, (px, py))

    def _draw_skeleton_fast(self, frame, pose, w, h):
        """Draw skeleton directly on frame - minimal"""
        kps = pose.keypoints
        conf = 0.3

        # Just draw key points and main lines
        connections = [
            (5, 6),   # shoulders
            (5, 11),  # left body
            (6, 12),  # right body
            (11, 12), # hips
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]

        for i, j in connections:
            if kps[i].confidence >= conf and kps[j].confidence >= conf:
                x1, y1 = int(kps[i].x * w), int(kps[i].y * h)
                x2, y2 = int(kps[j].x * w), int(kps[j].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw keypoints
        for kp in kps:
            if kp.confidence >= conf:
                x, y = int(kp.x * w), int(kp.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def cleanup(self):
        print("\nCleaning up...")
        pygame.quit()
        if self.camera:
            self.camera.release()
        if self.alert_system:
            self.alert_system.shutdown()
        rt = time.time() - self.start_time
        if self.frame_count > 0:
            print(f"{self.frame_count} frames, {self.frame_count/rt:.1f} FPS")


def main():
    app = FallDetectionDisplay()
    if app.initialize():
        app.run()

if __name__ == "__main__":
    main()
