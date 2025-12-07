# Elderly Fall Detection on Ethos-U NPU

Edge-based real-time fall detection and activity monitoring system for elderly care, running on NXP i.MX93 FRDM with Arm Ethos-U65 NPU acceleration.

![Platform](https://img.shields.io/badge/Platform-NXP%20i.MX93-blue)
![NPU](https://img.shields.io/badge/NPU-Ethos--U65-green)
![License](https://img.shields.io/badge/License-Apache%202.0-orange)

## Overview

This project implements a privacy-preserving, real-time fall detection system using pose estimation and activity classification. All inference runs on-device using the Ethos-U65 NPU, ensuring low latency and no cloud dependency.

### Key Features

- **Real-time pose estimation** using MoveNet Lightning quantized for NPU
- **Fall detection algorithm** with multi-factor analysis (body angle, velocity, bounding box)
- **Activity classification**: Standing, Sitting, Lying Down, Walking, Fallen
- **Inactivity monitoring** with configurable timeout alerts
- **Multiple display modes**: Framebuffer (30+ FPS), PyGame, Web Dashboard
- **NPU acceleration**: ~8ms inference time on Ethos-U65
- **Privacy-first**: All processing on-device, no video streaming

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Board | NXP i.MX93 FRDM |
| NPU | Arm Ethos-U65 |
| Camera | USB Webcam (640x480) or MIPI CSI |
| Display | HDMI Monitor (optional) |
| Memory | 2GB+ RAM |

## Software Requirements

- Linux (Yocto-based BSP for i.MX93)
- Python 3.10+
- TensorFlow Lite Runtime
- OpenCV
- Ethos-U delegate library

## Project Structure

```
elderly_fall_detection/
├── models/
│   └── movenet_lightning_int8_vela.tflite    # NPU-optimized model
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration and constants
│   ├── camera_capture.py          # Camera interface
│   ├── pose_estimator.py          # MoveNet inference with NPU
│   ├── fall_detector.py           # Fall detection algorithm
│   ├── activity_monitor.py        # Activity tracking
│   └── alert_system.py            # Alert management
├── app/
│   ├── __init__.py
│   └── main.py                    # CLI application
├── gui/
│   ├── fb_display.py              # Framebuffer display (fastest)
│   └── pygame_display.py          # PyGame display
├── web/
│   ├── server.py                  # Flask web dashboard
│   ├── API.md                     # REST API documentation
│   ├── templates/
│   │   └── dashboard.html
│   └── static/
│       ├── css/dashboard.css
│       └── js/dashboard.js
├── test_system.py                 # System verification
└── README.md
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/fidel-makatia/Elderly_Fall_Detection_on_ETHOS_NPU.git
cd Elderly_Fall_Detection_on_ETHOS_NPU
```

### 2. Install Dependencies

**On NXP Board:**
```bash
pip3 install numpy opencv-python-headless flask flask-socketio
```

**For Development (Mac/Linux):**
```bash
pip3 install -r requirements.txt
```

### 3. Verify NPU

```bash
ls /dev/ethosu*          # Should show /dev/ethosu0
ls /usr/lib/libethosu*   # Should show delegate library
```

## Usage

### Option 1: Framebuffer Display (Recommended - 30+ FPS)

```bash
killall weston  # Free the display
cd /home/root/elderly_fall_detection
python3 gui/fb_display.py
```

### Option 2: CLI Mode (Headless)

```bash
python3 app/main.py --no-display
```

### Option 3: Web Dashboard

```bash
python3 web/server.py
# Access: http://<board-ip>:5000
```

### Option 4: PyGame Display

```bash
export SDL_VIDEODRIVER=wayland
export WAYLAND_DISPLAY=wayland-1
export XDG_RUNTIME_DIR=/run/user/0
python3 gui/pygame_display.py
```

## Model Training & Optimization

### Step 1: Get Base Model

Download MoveNet Lightning from TensorFlow Hub:

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load MoveNet Lightning
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Get concrete function
concrete_func = model.signatures['serving_default']

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

# Representative dataset for quantization
def representative_dataset():
    for _ in range(100):
        data = np.random.randint(0, 255, (1, 192, 192, 3), dtype=np.uint8)
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

with open('movenet_lightning_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Step 2: Compile for Ethos-U NPU

Install Vela compiler:

```bash
pip3 install ethos-u-vela
```

Compile model for Ethos-U65:

```bash
vela movenet_lightning_int8.tflite \
    --accelerator-config ethos-u65-256 \
    --config vela.ini \
    --output-dir models/
```

Example `vela.ini`:

```ini
[System_Config.Ethos_U65_High_End]
macs_per_cc = 8
cmd_stream_version = 1
memory_mode = Shared_Sram

[Memory_Mode.Shared_Sram]
const_mem_area = Sram
arena_mem_area = Sram
cache_mem_area = Sram
arena_cache_size = 2097152
```

### Step 3: Verify Model

```bash
vela --show-cpu-operations movenet_lightning_int8_vela.tflite
```

Output shows operations delegated to NPU vs CPU.

## Fall Detection Algorithm

The fall detector uses multiple factors:

| Factor | Threshold | Weight |
|--------|-----------|--------|
| Body Angle | > 50° from vertical | 30% |
| Vertical Velocity | > 0.15 units/frame | 25% |
| Height/Width Ratio | < 0.8 (horizontal) | 20% |
| Upper/Lower Y Distance | < 0.1 (same level) | 15% |
| X-axis Spread | > 0.4 (sprawled) | 10% |

A fall is confirmed when:
- `(angle > threshold AND velocity > threshold)` OR
- `(angle > threshold AND y_diff < threshold)` OR
- `3+ factors exceeded`

And persists for `confirm_frames` (default: 3) consecutive frames.

### Tuning Parameters

Edit `src/config.py`:

```python
@dataclass
class FallDetectionConfig:
    confidence_threshold: float = 0.3    # Keypoint confidence
    angle_threshold: float = 50.0        # Degrees from vertical
    velocity_threshold: float = 0.15     # Fall speed
    confirm_frames: int = 3              # Frames to confirm fall
    recovery_frames: int = 10            # Frames to confirm recovery
```

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current system state |
| `/api/statistics` | GET | Session statistics |
| `/api/alerts` | GET | Alert history |
| `/api/reset` | POST | Reset detection |
| `/api/test-alert` | POST | Trigger test alert |
| `/video_feed` | GET | MJPEG stream |

### WebSocket Events

```javascript
// Connect
const socket = io('http://<board-ip>:5000');

// Receive status updates
socket.on('status_update', (data) => {
    console.log(data.state, data.metrics.fps);
});

// Receive alerts
socket.on('alert', (data) => {
    if (data.type === 'fall') {
        // Handle fall alert
    }
});
```

See `web/API.md` for complete documentation.

## Performance

| Metric | Value |
|--------|-------|
| Inference Time | ~8ms (NPU) |
| End-to-End Latency | ~25ms |
| Frame Rate | 30+ FPS (framebuffer) |
| Power Consumption | ~2W (typical) |
| Model Size | 2.5MB (Vela optimized) |

## Troubleshooting

### Camera Not Found

```bash
ls /dev/video*
# If empty, connect USB camera and check:
dmesg | tail -20
```

### NPU Not Detected

```bash
# Check device
ls /dev/ethosu0

# Check delegate
ls /usr/lib/libethosu_delegate.so

# Verify in Python
python3 -c "import tflite_runtime.interpreter as tflite; print('OK')"
```

### Low FPS

1. Use framebuffer mode (`fb_display.py`)
2. Kill Weston: `killall weston`
3. Reduce camera resolution in `config.py`
4. Verify NPU is active (check logs for "NPU delegate loaded")

### Fall Not Detecting

1. Check keypoint confidence (need 5+ valid keypoints)
2. Verify body is fully visible in frame
3. Adjust thresholds in `config.py`
4. Monitor angle values during fall

## Use Case

**Edge-Based Elderly Fall Detection and Activity Monitoring**

This Arm-sponsored project demonstrates how Arm edge AI solutions address healthcare needs by providing privacy-preserving, real-time fall detection for the elderly. Using the NXP i.MX 8M Plus platform with Ethos-U65 NPU, advanced pose estimation runs entirely on-device, instantly alerting caregivers to falls or inactivity without streaming sensitive video to the cloud.

### Problem

Falls are the leading cause of injury-related deaths among older adults, with most incidents unwitnessed or discovered too late. Traditional wearable alarms often go unused or fail to capture context.

### Solution

A wide-angle camera monitors living spaces. Pose estimation (MoveNet) is quantized and optimized for the Ethos-U65 NPU. All inference is on-device, ensuring privacy and 24/7 reliability. When a fall or unusual inactivity is detected, instant alerts notify caregivers.

## License

Apache License 2.0

## Acknowledgments

- Arm Developer Program
- NXP Semiconductors
- TensorFlow Team (MoveNet model)
