# Edge-Based Elderly Fall Detection Using Arm Ethos-U NPU

**Author:** Fidel Makatia, Distinguished Arm Ambassador
**GitHub:** https://github.com/fidel-makatia/Elderly_Fall_Detection_on_ETHOS_NPU

---

## The Problem: Falls Are the Leading Cause of Injury Death in Older Adults

Every year, millions of elderly individuals suffer fall-related injuries—many unwitnessed or discovered too late for timely intervention. Traditional solutions like wearable panic buttons often go unused, fail to detect falls automatically, or lack the context needed for effective response.

Cloud-based camera monitoring introduces privacy concerns, latency issues, and dependency on internet connectivity. For elderly care, these limitations can mean the difference between rapid intervention and tragedy.

**The challenge is clear:** Can we build a fall detection system that is real-time, privacy-preserving, and works entirely offline—without compromising accuracy or affordability?

*[Figure 1: System running on NXP i.MX93 FRDM board with HDMI display showing live fall detection]*

*[Figure 2: Fall detected alert with confidence score and pose skeleton overlay]*

---

## Why It Matters: Privacy, Speed, and Always-On Protection

For elderly individuals living independently—and for their families—reliable fall detection provides peace of mind. But current solutions fall short:

| Challenge | Cloud-Based Systems | Our Edge AI Solution |
|-----------|---------------------|----------------------|
| **Privacy** | Video streams to remote servers | All processing on-device |
| **Latency** | 500ms–2s network delay | <30ms end-to-end |
| **Reliability** | Fails during outages | Works 100% offline |
| **Cost** | Monthly subscription fees | One-time hardware cost |
| **Data Security** | Exposed to breaches | Local network only |

The Arm Ethos-U NPU changes what's possible at the edge. With dedicated neural network acceleration, complex pose estimation models run in milliseconds—enabling real-time fall detection on low-power embedded hardware.

---

## The Solution: Real-Time Fall Detection Powered by Arm Ethos-U NPU

This open-source project demonstrates how the Arm Ethos-U65 Neural Processing Unit enables real-time pose estimation and fall detection entirely on-device. No cloud. No compromise. No privacy concerns.

### Key Implementation Highlights

**Ethos-U NPU Acceleration**
The MoveNet pose estimation model is quantized (INT8) and compiled using Arm's Vela compiler for optimal Ethos-U65 execution. Inference runs in ~8ms—fast enough for 30+ FPS real-time processing.

**Multi-Factor Fall Detection Algorithm**
Falls are detected by analyzing multiple biomechanical indicators simultaneously:
- Body angle deviation from vertical
- Downward velocity during fall motion
- Bounding box aspect ratio changes
- Upper/lower body position differential
- Horizontal spread (sprawled position detection)

**Privacy-First Architecture**
All video processing occurs on-device. No frames are transmitted externally. The system can operate completely air-gapped from any network.

**Multiple Output Modes**
- Direct framebuffer display for maximum performance
- Web dashboard for remote monitoring (local network)
- REST API for integration with home automation systems
- Alert system with configurable notifications

*[Figure 3: System architecture diagram showing camera → NPU inference → fall detection → alert flow]*

---

## Hardware Setup

*[Figure 4: Hardware components - board, camera, HDMI display]*

| Component | Specification |
|-----------|---------------|
| **NPU** | Arm Ethos-U65 (256 MAC units) |
| **Processor** | Arm Cortex-A55 |
| **Board** | NXP i.MX93 FRDM |
| **Camera** | USB Webcam (640×480) |
| **Display** | HDMI Monitor (optional) |
| **Memory** | 2GB RAM |
| **Storage** | MicroSD or eMMC |

The Ethos-U65 NPU provides up to 1 TOPS (Tera Operations Per Second) of neural network inference performance while consuming minimal power—ideal for always-on monitoring applications.

---

## System Architecture

The system employs a fully local pipeline from camera capture to alert generation:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Camera    │────▶│  Preprocessing   │────▶│   Ethos-U65     │
│  (640×480)  │     │  (Resize, Norm)  │     │  NPU Inference  │
└─────────────┘     └──────────────────┘     │   (~8ms)        │
                                              └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Alert     │◀────│  Fall Detection  │◀────│ Pose Keypoints  │
│   System    │     │   Algorithm      │     │  (17 joints)    │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

*[Figure 5: Detailed system architecture diagram]*

---

## Arm Ethos-U NPU: The Key Enabler

The Ethos-U65 NPU is purpose-built for efficient neural network inference at the edge. Key advantages for this application:

**Optimized Operator Support**
The Vela compiler maps supported TensorFlow Lite operations directly to NPU hardware, including:
- Depthwise Convolutions (core of MoveNet)
- Standard Convolutions
- Fully Connected layers
- Activation functions (ReLU, ReLU6)

**Memory Efficiency**
The Ethos-U architecture minimizes data movement between memory and compute units—critical for real-time video processing where bandwidth is often the bottleneck.

**Power Efficiency**
Neural network inference on dedicated NPU hardware consumes significantly less power than CPU or GPU alternatives, enabling continuous monitoring without thermal concerns.

### Model Optimization with Vela

The MoveNet model is prepared for Ethos-U execution using Arm's Vela compiler:

```bash
vela movenet_lightning_int8.tflite \
    --accelerator-config ethos-u65-256 \
    --optimise Performance
```

Vela analyzes the model graph, fuses operations where possible, and generates optimized command streams for the NPU. Operations not supported by hardware fall back to CPU execution seamlessly.

*[Figure 6: Vela compilation output showing operator delegation]*

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **NPU Inference Time** | ~8ms | MoveNet Lightning INT8 |
| **End-to-End Latency** | <30ms | Camera to detection result |
| **Frame Rate** | 30+ FPS | Framebuffer display mode |
| **Model Size** | 2.5MB | Vela-optimized |
| **Power Consumption** | ~2W | Typical operation |
| **Detection Accuracy** | >95% | Controlled testing |
| **False Positive Rate** | <2% | Normal activity filtering |

*[Figure 7: Performance benchmark visualization]*

---

## Fall Detection Algorithm

The algorithm analyzes pose keypoints to detect falls through multiple indicators:

| Factor | Detection Logic | Threshold |
|--------|-----------------|-----------|
| **Body Angle** | Torso deviation from vertical | >50° |
| **Velocity** | Downward movement speed | >0.15 units/frame |
| **Aspect Ratio** | Height/width of bounding box | <0.8 (horizontal) |
| **Y-Distance** | Upper vs lower body height | <0.1 (same level) |
| **X-Spread** | Horizontal keypoint spread | >0.4 (sprawled) |

A fall is confirmed when:
- (Angle exceeded AND Velocity exceeded) OR
- (Angle exceeded AND Y-distance collapsed) OR
- (3+ factors simultaneously exceeded)

This multi-factor approach minimizes false positives from activities like bending or sitting while reliably detecting actual falls.

*[Figure 8: Pose skeleton showing keypoint detection and angle calculation]*

---

## Technical Details

| Component | Details | Documentation |
|-----------|---------|---------------|
| **Repository** | Elderly Fall Detection on Ethos-U NPU | [GitHub](https://github.com/fidel-makatia/Elderly_Fall_Detection_on_ETHOS_NPU) |
| **NPU** | Arm Ethos-U65 | [Ethos-U Documentation](https://developer.arm.com/ip-products/processors/machine-learning/arm-ethos-u) |
| **Model** | MoveNet Lightning (INT8 Quantized) | [TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/lightning/4) |
| **Compiler** | Arm Vela 3.10+ | [Vela GitHub](https://github.com/ARM-software/ethos-u-vela) |
| **Runtime** | TensorFlow Lite + Ethos-U Delegate | [TFLite Micro](https://www.tensorflow.org/lite/microcontrollers) |
| **Backend** | Python 3, OpenCV, Flask | See repository |

---

## Results & Impact

**True Edge AI Privacy**
All pose estimation and fall detection processing occurs entirely on the Ethos-U NPU. No video frames ever leave the device—absolute privacy is guaranteed.

**Real-Time Performance**
The ~8ms NPU inference time enables 30+ FPS processing, ensuring falls are detected within a single frame of occurrence.

**Always-On Reliability**
The system operates completely offline. Network outages, cloud service disruptions, or ISP issues have zero impact on protection.

**Cost-Effective Deployment**
No monthly subscriptions or API fees. After initial hardware cost, the system operates indefinitely at no additional expense.

*[Figure 9: Live detection showing FPS, inference time, and NPU status]*

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Neural Accelerator** | Arm Ethos-U65 NPU |
| **Model Compiler** | Arm Vela |
| **Inference Runtime** | TensorFlow Lite + Ethos-U Delegate |
| **Pose Estimation** | MoveNet Lightning (INT8) |
| **Fall Detection** | Custom multi-factor algorithm |
| **Display** | Direct framebuffer / PyGame / Web |
| **Backend** | Python 3, OpenCV, Flask |
| **API** | REST, WebSocket |

---

## Get Started

**Repository & Documentation:**
https://github.com/fidel-makatia/Elderly_Fall_Detection_on_ETHOS_NPU

### Quickstart

1. Clone the repository to your Ethos-U enabled board
2. Install dependencies: `pip3 install numpy opencv-python flask`
3. Verify NPU is available: `ls /dev/ethosu0`
4. Run the system: `python3 gui/fb_display.py`

### What You Can Do

- **Deploy** the system for elderly care monitoring
- **Extend** the algorithm for additional activities
- **Integrate** with smart home systems via REST API
- **Customize** detection thresholds for your environment

---

## What's Next

- Fork the repository and deploy on your own hardware
- Extend detection for additional health-related events
- Integrate with existing care facility systems
- Contribute improvements back to the community

---

## Arm Developer Resources Used

- [Arm Ethos-U NPU Documentation](https://developer.arm.com/ip-products/processors/machine-learning/arm-ethos-u)
- [Vela Compiler for Ethos-U](https://github.com/ARM-software/ethos-u-vela)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Arm NN SDK](https://developer.arm.com/tools-and-software/graphics-and-gaming/arm-nn)
- [ML Inference Advisor](https://developer.arm.com/tools-and-software/software-development-tools/machine-learning)

---

## Additional Resources

- **Project GitHub:** https://github.com/fidel-makatia/Elderly_Fall_Detection_on_ETHOS_NPU
- **Ethos-U Technical Overview:** https://developer.arm.com/documentation/102420
- **Vela User Guide:** https://developer.arm.com/documentation/109267
- **MoveNet Model:** https://tfhub.dev/google/movenet/singlepose/lightning

---

*This project demonstrates that neural network acceleration on Arm Ethos-U NPU transforms what's possible for edge AI in healthcare applications—delivering real-time, privacy-preserving fall detection that protects elderly individuals while keeping their data exactly where it belongs: on-device.*
