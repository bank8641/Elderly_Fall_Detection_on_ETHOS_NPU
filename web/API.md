# Fall Detection System - REST API Documentation

## Overview

The Fall Detection System provides a RESTful API for integrating with external systems, mobile apps, and IoT devices. All endpoints return JSON responses.

**Base URL:** `http://<device-ip>:5000`

---

## Endpoints

### System Status

#### GET `/api/status`

Returns the current system state and real-time metrics.

**Response:**
```json
{
    "state": "running",
    "activity": "standing",
    "metrics": {
        "fps": 28.5,
        "inference_ms": 35.2,
        "npu_active": true,
        "uptime": 3600
    },
    "detection": {
        "is_fall": false,
        "fall_confidence": 0.0,
        "body_angle": 15.3,
        "person_detected": true,
        "keypoint_count": 14
    }
}
```

**State Values:**
- `initializing` - System starting up
- `running` - Normal operation
- `fall_detected` - Active fall alert
- `inactive` - Person inactivity detected
- `error` - System error

**Activity Values:**
- `standing`, `sitting`, `lying_down`, `walking`, `fallen`, `not_detected`, `unknown`

---

### Statistics

#### GET `/api/statistics`

Returns comprehensive session statistics.

**Response:**
```json
{
    "uptime_seconds": 7200,
    "total_frames": 216000,
    "average_fps": 30.0,
    "average_inference_ms": 33.5,
    "npu_enabled": true,
    "total_falls_detected": 2,
    "total_alerts": 3,
    "alerts_by_type": {
        "fall": 2,
        "inactivity": 1
    }
}
```

---

### Alert Management

#### GET `/api/alerts`

Returns alert history (last 100 alerts).

**Response:**
```json
[
    {
        "id": "FALL-0001",
        "timestamp": 1702345678.123,
        "alert_type": "fall",
        "priority": "critical",
        "message": "Fall detected in Living Room",
        "details": {
            "fall_type": "rapid_fall",
            "confidence": 0.92,
            "body_angle": 75.3,
            "velocity": 0.45
        },
        "acknowledged": false
    }
]
```

**Priority Levels:**
- `critical` - Immediate attention required (falls)
- `warning` - Important but not urgent (inactivity)
- `info` - Informational (test alerts)

---

#### POST `/api/alerts/{alert_id}/acknowledge`

Acknowledge an alert.

**Response:**
```json
{
    "success": true
}
```

---

#### POST `/api/test-alert`

Trigger a test alert for integration testing.

**Response:**
```json
{
    "success": true,
    "alert_id": "TEST-0001"
}
```

---

### System Control

#### POST `/api/reset`

Reset the detection state (clears fall detection, resets counters).

**Response:**
```json
{
    "success": true,
    "message": "Detection state reset"
}
```

---

### Configuration

#### GET `/api/config`

Returns current system configuration.

**Response:**
```json
{
    "fall_detection": {
        "confidence_threshold": 0.3,
        "angle_threshold": 50.0,
        "velocity_threshold": 0.15,
        "confirm_frames": 3
    },
    "activity": {
        "inactivity_timeout": 300.0,
        "movement_threshold": 0.02
    },
    "alert": {
        "location_name": "Living Room",
        "cooldown": 30.0
    }
}
```

---

### Video Stream

#### GET `/video_feed`

MJPEG video stream with pose overlay and status indicators.

**Content-Type:** `multipart/x-mixed-replace; boundary=frame`

**Usage in HTML:**
```html
<img src="http://<device-ip>:5000/video_feed" />
```

---

## WebSocket Events

Connect to the WebSocket server for real-time updates.

**Connection:**
```javascript
const socket = io('http://<device-ip>:5000');
```

### Events

#### `status_update`
Broadcast every second with current system status.

```javascript
socket.on('status_update', (data) => {
    console.log(data.state, data.metrics.fps);
});
```

#### `alert`
Broadcast when an alert is triggered.

```javascript
socket.on('alert', (data) => {
    if (data.priority === 'critical') {
        // Handle fall alert
    }
});
```

#### `request_status`
Request immediate status update.

```javascript
socket.emit('request_status');
```

---

## Integration Examples

### Python Client

```python
import requests
import time

BASE_URL = "http://192.168.1.100:5000"

# Get current status
response = requests.get(f"{BASE_URL}/api/status")
status = response.json()

if status['detection']['is_fall']:
    print(f"FALL DETECTED! Confidence: {status['detection']['fall_confidence']}")

# Send test alert
requests.post(f"{BASE_URL}/api/test-alert")
```

### Node.js Client

```javascript
const io = require('socket.io-client');
const socket = io('http://192.168.1.100:5000');

socket.on('alert', (alert) => {
    if (alert.type === 'fall') {
        // Send notification to caregiver
        sendPushNotification(alert.message);
    }
});

socket.on('status_update', (status) => {
    console.log(`FPS: ${status.metrics.fps}, Person: ${status.detection.person_detected}`);
});
```

### cURL

```bash
# Get status
curl http://192.168.1.100:5000/api/status

# Send test alert
curl -X POST http://192.168.1.100:5000/api/test-alert

# Reset detection
curl -X POST http://192.168.1.100:5000/api/reset
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request
- `404` - Resource not found
- `500` - Server error

Error response format:
```json
{
    "error": "Error message",
    "code": "ERROR_CODE"
}
```

---

## Rate Limiting

The API does not currently implement rate limiting, but clients should:
- Avoid polling `/api/status` more than once per second
- Use WebSocket for real-time updates instead of polling
- Cache `/api/config` as it rarely changes

---

## Security Considerations

For production deployments:

1. **Network Security:** Run behind a reverse proxy (nginx) with HTTPS
2. **Authentication:** Implement API key or JWT authentication
3. **Firewall:** Restrict access to trusted IP addresses
4. **CORS:** Configure allowed origins for web clients

Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name fall-detection.local;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
