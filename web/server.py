#!/usr/bin/env python3
"""
Elderly Fall Detection - Web Dashboard Server
NXP i.MX93 FRDM + Arm Ethos-U65 NPU

A professional-grade web interface for real-time fall detection monitoring.
Accessible via http://localhost:5000

Author: Arm Edge AI Healthcare Initiative
License: Apache 2.0
"""

import sys
import os
import time
import json
import base64
import threading
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    CAMERA_DEVICE, SystemState, ActivityState,
    FALL_CONFIG, ACTIVITY_CONFIG, ALERT_CONFIG
)
from camera_capture import CameraCapture
from pose_estimator import PoseEstimator, visualize_pose
from fall_detector import FallDetector, FallEvent
from activity_monitor import ActivityMonitor
from alert_system import AlertSystem


# =============================================================================
# Application Configuration
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arm-edge-ai-healthcare-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    fps: float = 0.0
    inference_time_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    npu_active: bool = False
    uptime_seconds: float = 0.0


@dataclass
class AlertRecord:
    """Persistent alert record for history"""
    id: str
    timestamp: float
    alert_type: str  # 'fall', 'inactivity', 'test'
    priority: str    # 'critical', 'warning', 'info'
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False


# =============================================================================
# Detection Engine
# =============================================================================

class DetectionEngine:
    """
    Core detection engine that integrates all components.
    Runs in a separate thread and broadcasts updates via WebSocket.
    """

    def __init__(self, camera_device: str = CAMERA_DEVICE):
        self.camera_device = camera_device

        # Components
        self.camera: Optional[CameraCapture] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.fall_detector: Optional[FallDetector] = None
        self.activity_monitor: Optional[ActivityMonitor] = None
        self.alert_system: Optional[AlertSystem] = None

        # State
        self.state = SystemState.INITIALIZING
        self.activity_state = ActivityState.UNKNOWN
        self.metrics = SystemMetrics()

        # Runtime
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time = 0.0
        self.frame_count = 0

        # Alert history (last 100 alerts)
        self.alert_history: deque = deque(maxlen=100)
        self.alert_counter = 0

        # Latest frame for streaming
        self._latest_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        # Detection state
        self._current_detection = {
            'is_fall': False,
            'fall_confidence': 0.0,
            'body_angle': 0.0,
            'person_detected': False,
            'keypoint_count': 0
        }

    def initialize(self) -> bool:
        """Initialize all detection components"""
        try:
            print("[Engine] Initializing camera...")
            self.camera = CameraCapture(device=self.camera_device)
            if not self.camera.initialize():
                print("[Engine] Camera initialization failed")
                return False

            print("[Engine] Initializing pose estimator...")
            self.pose_estimator = PoseEstimator()
            if not self.pose_estimator.initialize():
                print("[Engine] Pose estimator initialization failed")
                return False

            print("[Engine] Initializing fall detector...")
            self.fall_detector = FallDetector()
            self.fall_detector.register_fall_callback(self._on_fall_detected)

            print("[Engine] Initializing activity monitor...")
            self.activity_monitor = ActivityMonitor()
            self.activity_monitor.register_inactivity_callback(self._on_inactivity_detected)

            print("[Engine] Initializing alert system...")
            self.alert_system = AlertSystem()
            self.alert_system.initialize()

            self.metrics.npu_active = self.pose_estimator.is_using_npu()
            self.state = SystemState.RUNNING

            print("[Engine] All components initialized successfully")
            return True

        except Exception as e:
            print(f"[Engine] Initialization error: {e}")
            self.state = SystemState.ERROR
            return False

    def start(self):
        """Start the detection loop in a background thread"""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("[Engine] Detection loop started")

    def stop(self):
        """Stop the detection loop"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._cleanup()
        print("[Engine] Detection loop stopped")

    def _detection_loop(self):
        """Main detection loop"""
        fps_update_time = time.time()
        fps_frame_count = 0

        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Run pose estimation
                preprocessed = self.camera.get_preprocessed_frame()
                pose_result = None
                if preprocessed is not None:
                    pose_result = self.pose_estimator.estimate(preprocessed)

                # Process detection
                detection_state = None
                if pose_result:
                    detection_state = self.fall_detector.process(pose_result)
                    self._current_detection['is_fall'] = detection_state.is_fall_detected
                    self._current_detection['fall_confidence'] = detection_state.fall_confidence
                    self._current_detection['body_angle'] = detection_state.body_angle
                    self._current_detection['keypoint_count'] = len(pose_result.get_valid_keypoints(0.3))
                    self.activity_state = detection_state.activity_state

                    if detection_state.is_fall_detected:
                        self.state = SystemState.FALL_DETECTED
                    elif self.state == SystemState.FALL_DETECTED:
                        self.state = SystemState.RUNNING

                # Check person detection
                person_detected = pose_result and len(pose_result.get_valid_keypoints(0.3)) >= 5
                self._current_detection['person_detected'] = person_detected

                # Activity monitoring
                activity = self.activity_monitor.process(pose_result, person_detected)

                # Update metrics
                if pose_result:
                    self.metrics.inference_time_ms = pose_result.inference_time_ms

                # Render frame with overlay
                display_frame = self._render_frame(frame, pose_result, detection_state)

                # Encode for streaming
                with self._frame_lock:
                    _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self._latest_frame = buffer.tobytes()

                # Update FPS
                self.frame_count += 1
                fps_frame_count += 1

                now = time.time()
                if now - fps_update_time >= 1.0:
                    self.metrics.fps = fps_frame_count / (now - fps_update_time)
                    self.metrics.uptime_seconds = now - self.start_time
                    fps_frame_count = 0
                    fps_update_time = now

                    # Broadcast status update
                    self._broadcast_status()

            except Exception as e:
                print(f"[Engine] Detection loop error: {e}")
                time.sleep(0.1)

    def _render_frame(self, frame: np.ndarray, pose, detection_state) -> np.ndarray:
        """Render frame with pose overlay and status indicators"""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw pose skeleton
        if pose:
            display = visualize_pose(display, pose)

        # Top status bar
        cv2.rectangle(display, (0, 0), (w, 60), (20, 20, 20), -1)

        # Title
        cv2.putText(display, "FALL DETECTION SYSTEM", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # NPU indicator
        npu_text = "NPU" if self.metrics.npu_active else "CPU"
        npu_color = (0, 255, 0) if self.metrics.npu_active else (0, 165, 255)
        cv2.putText(display, npu_text, (w - 60, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, npu_color, 2)

        # Metrics
        metrics_text = f"FPS: {self.metrics.fps:.1f} | Inference: {self.metrics.inference_time_ms:.0f}ms"
        cv2.putText(display, metrics_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Activity state
        if detection_state:
            activity_text = f"Activity: {detection_state.activity_state.value.upper()}"
            cv2.putText(display, activity_text, (w - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Fall alert overlay
        if detection_state and detection_state.is_fall_detected:
            # Red border
            cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 0, 255), 8)

            # Alert banner
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            cv2.putText(display, "FALL DETECTED", (w//2 - 140, h//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            conf_text = f"Confidence: {detection_state.fall_confidence:.0%}"
            cv2.putText(display, conf_text, (w//2 - 80, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, timestamp, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display

    def _on_fall_detected(self, event: FallEvent):
        """Handle fall detection event"""
        self.alert_counter += 1

        alert = AlertRecord(
            id=f"FALL-{self.alert_counter:04d}",
            timestamp=event.timestamp,
            alert_type='fall',
            priority='critical',
            message=f"Fall detected in {ALERT_CONFIG.location_name}",
            details={
                'fall_type': event.fall_type,
                'confidence': event.confidence,
                'body_angle': event.body_angle,
                'velocity': event.velocity
            }
        )

        self.alert_history.append(alert)
        self._broadcast_alert(alert)

    def _on_inactivity_detected(self, event):
        """Handle inactivity detection event"""
        self.alert_counter += 1

        alert = AlertRecord(
            id=f"INACT-{self.alert_counter:04d}",
            timestamp=event.timestamp,
            alert_type='inactivity',
            priority='warning',
            message=f"Extended inactivity in {ALERT_CONFIG.location_name}",
            details={
                'duration_minutes': event.duration / 60
            }
        )

        self.alert_history.append(alert)
        self._broadcast_alert(alert)

    def _broadcast_status(self):
        """Broadcast current status via WebSocket"""
        status = {
            'state': self.state.value,
            'activity': self.activity_state.value,
            'metrics': {
                'fps': round(self.metrics.fps, 1),
                'inference_ms': round(self.metrics.inference_time_ms, 1),
                'npu_active': self.metrics.npu_active,
                'uptime': int(self.metrics.uptime_seconds)
            },
            'detection': self._current_detection,
            'timestamp': time.time()
        }
        socketio.emit('status_update', status)

    def _broadcast_alert(self, alert: AlertRecord):
        """Broadcast alert via WebSocket"""
        alert_data = {
            'id': alert.id,
            'timestamp': alert.timestamp,
            'type': alert.alert_type,
            'priority': alert.priority,
            'message': alert.message,
            'details': alert.details
        }
        socketio.emit('alert', alert_data)

    def get_frame(self) -> Optional[bytes]:
        """Get latest encoded frame"""
        with self._frame_lock:
            return self._latest_frame

    def get_alert_history(self) -> List[Dict]:
        """Get alert history as list of dicts"""
        return [asdict(a) for a in self.alert_history]

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        fall_events = self.fall_detector.get_fall_events() if self.fall_detector else []

        return {
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'total_frames': self.frame_count,
            'average_fps': self.frame_count / (time.time() - self.start_time) if self.start_time else 0,
            'average_inference_ms': self.pose_estimator.get_average_inference_time() if self.pose_estimator else 0,
            'npu_enabled': self.metrics.npu_active,
            'total_falls_detected': len(fall_events),
            'total_alerts': len(self.alert_history),
            'alerts_by_type': {
                'fall': sum(1 for a in self.alert_history if a.alert_type == 'fall'),
                'inactivity': sum(1 for a in self.alert_history if a.alert_type == 'inactivity')
            }
        }

    def reset_detection(self):
        """Reset detection state"""
        if self.fall_detector:
            self.fall_detector.reset()
        if self.activity_monitor:
            self.activity_monitor.reset()
        self.state = SystemState.RUNNING
        self._current_detection = {
            'is_fall': False,
            'fall_confidence': 0.0,
            'body_angle': 0.0,
            'person_detected': False,
            'keypoint_count': 0
        }

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def _cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()
        if self.alert_system:
            self.alert_system.shutdown()


# =============================================================================
# Global Detection Engine Instance
# =============================================================================

engine = DetectionEngine()


# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get current system status"""
    return jsonify({
        'state': engine.state.value,
        'activity': engine.activity_state.value,
        'metrics': {
            'fps': round(engine.metrics.fps, 1),
            'inference_ms': round(engine.metrics.inference_time_ms, 1),
            'npu_active': engine.metrics.npu_active,
            'uptime': int(engine.metrics.uptime_seconds)
        },
        'detection': engine._current_detection
    })


@app.route('/api/statistics')
def api_statistics():
    """Get comprehensive statistics"""
    return jsonify(engine.get_statistics())


@app.route('/api/alerts')
def api_alerts():
    """Get alert history"""
    return jsonify(engine.get_alert_history())


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def api_acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    success = engine.acknowledge_alert(alert_id)
    return jsonify({'success': success})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection state"""
    engine.reset_detection()
    return jsonify({'success': True, 'message': 'Detection state reset'})


@app.route('/api/test-alert', methods=['POST'])
def api_test_alert():
    """Send a test alert"""
    engine.alert_counter += 1
    alert = AlertRecord(
        id=f"TEST-{engine.alert_counter:04d}",
        timestamp=time.time(),
        alert_type='test',
        priority='info',
        message='Test alert from dashboard',
        details={'source': 'manual_test'}
    )
    engine.alert_history.append(alert)
    engine._broadcast_alert(alert)
    return jsonify({'success': True, 'alert_id': alert.id})


@app.route('/api/config')
def api_config():
    """Get current configuration"""
    return jsonify({
        'fall_detection': {
            'confidence_threshold': FALL_CONFIG.confidence_threshold,
            'angle_threshold': FALL_CONFIG.angle_threshold,
            'velocity_threshold': FALL_CONFIG.velocity_threshold,
            'confirm_frames': FALL_CONFIG.confirm_frames
        },
        'activity': {
            'inactivity_timeout': ACTIVITY_CONFIG.inactivity_timeout,
            'movement_threshold': ACTIVITY_CONFIG.movement_threshold
        },
        'alert': {
            'location_name': ALERT_CONFIG.location_name,
            'cooldown': ALERT_CONFIG.alert_cooldown
        }
    })


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint"""
    def generate():
        while True:
            frame = engine.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# WebSocket Events
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"[WebSocket] Client connected")
    emit('connected', {'message': 'Connected to Fall Detection System'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"[WebSocket] Client disconnected")


@socketio.on('request_status')
def handle_status_request():
    """Handle status request"""
    engine._broadcast_status()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Start the web server and detection engine"""
    import argparse

    parser = argparse.ArgumentParser(description='Fall Detection Web Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--camera', default=CAMERA_DEVICE, help='Camera device')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("=" * 60)
    print("  Elderly Fall Detection System - Web Dashboard")
    print("  NXP i.MX93 FRDM + Arm Ethos-U65 NPU")
    print("=" * 60)
    print()

    # Initialize detection engine
    engine.camera_device = args.camera
    if not engine.initialize():
        print("Failed to initialize detection engine!")
        sys.exit(1)

    # Start detection loop
    engine.start()

    print()
    print(f"Dashboard URL: http://localhost:{args.port}")
    print(f"Video Stream:  http://localhost:{args.port}/video_feed")
    print(f"API Endpoint:  http://localhost:{args.port}/api/status")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()


if __name__ == '__main__':
    main()
