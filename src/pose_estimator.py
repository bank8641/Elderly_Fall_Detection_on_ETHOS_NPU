"""
Pose Estimator with NPU Acceleration
"""

import numpy as np
import time
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass

from config import MOVENET_MODEL_PATH, MOVENET_INPUT_SIZE, NPU_DELEGATE_PATH, USE_NPU, KEYPOINT_NAMES, Keypoint, SKELETON_EDGES

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


@dataclass
class KeypointData:
    index: int
    name: str
    x: float
    y: float
    confidence: float

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class PoseResult:
    keypoints: List[KeypointData]
    inference_time_ms: float
    timestamp: float

    def get_keypoint(self, kp: Keypoint) -> Optional[KeypointData]:
        idx = kp.value
        return self.keypoints[idx] if 0 <= idx < len(self.keypoints) else None

    def get_valid_keypoints(self, min_conf: float = 0.3) -> List[KeypointData]:
        return [kp for kp in self.keypoints if kp.confidence >= min_conf]

    def get_center_of_mass(self) -> Tuple[float, float]:
        valid = self.get_valid_keypoints()
        if not valid:
            return (0.5, 0.5)
        return (sum(kp.x for kp in valid) / len(valid), sum(kp.y for kp in valid) / len(valid))

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        valid = self.get_valid_keypoints()
        if not valid:
            return (0, 0, 1, 1)
        xs = [kp.x for kp in valid]
        ys = [kp.y for kp in valid]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_array(self) -> np.ndarray:
        arr = np.zeros((17, 3), dtype=np.float32)
        for kp in self.keypoints:
            arr[kp.index] = [kp.y, kp.x, kp.confidence]
        return arr


class PoseEstimator:
    def __init__(self, model_path: str = MOVENET_MODEL_PATH, use_npu: bool = USE_NPU):
        self.model_path = model_path
        self.use_npu = use_npu
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._using_npu = False
        self._inference_times = []

    def initialize(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"Model not found: {self.model_path}")
            return False

        print(f"Loading model: {self.model_path}")

        if self.use_npu and os.path.exists(NPU_DELEGATE_PATH) and os.path.exists('/dev/ethosu0'):
            try:
                print("Loading NPU delegate...")
                delegate = tflite.load_delegate(NPU_DELEGATE_PATH)
                self._interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[delegate],
                    num_threads=2
                )
                self._using_npu = True
                print("NPU delegate loaded!")
            except Exception as e:
                print(f"NPU failed: {e}, using CPU")
                self._interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=4)
        else:
            print("Using CPU inference")
            self._interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=4)

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        print(f"Input: {self._input_details[0]['shape']}, dtype: {self._input_details[0]['dtype']}")
        print(f"Output: {self._output_details[0]['shape']}")
        return True

    def estimate(self, image: np.ndarray) -> Optional[PoseResult]:
        if self._interpreter is None:
            return None

        try:
            if image.ndim == 4:
                image = image[0]
            if image.shape[:2] != (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE):
                import cv2
                image = cv2.resize(image, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))

            tensor = np.expand_dims(image, axis=0)
            input_dtype = self._input_details[0]['dtype']
            if input_dtype == np.uint8:
                tensor = tensor.astype(np.uint8)
            elif input_dtype == np.int8:
                tensor = tensor.astype(np.int8)
            else:
                tensor = tensor.astype(np.float32) / 255.0

            start = time.perf_counter()
            self._interpreter.set_tensor(self._input_details[0]['index'], tensor)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]['index'])
            inference_time = (time.perf_counter() - start) * 1000

            self._inference_times.append(inference_time)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)

            if output.ndim == 4:
                output = output[0, 0]
            elif output.ndim == 3:
                output = output[0]

            keypoints = []
            for i in range(17):
                y, x, conf = output[i]
                keypoints.append(KeypointData(i, KEYPOINT_NAMES[i], float(x), float(y), float(conf)))

            return PoseResult(keypoints, inference_time, time.time())

        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def get_average_inference_time(self) -> float:
        return sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0

    def is_using_npu(self) -> bool:
        return self._using_npu

    def release(self):
        self._interpreter = None


def visualize_pose(image: np.ndarray, pose: PoseResult, min_conf: float = 0.3) -> np.ndarray:
    import cv2
    img = image.copy()
    h, w = img.shape[:2]

    for kp1_enum, kp2_enum in SKELETON_EDGES:
        kp1 = pose.get_keypoint(kp1_enum)
        kp2 = pose.get_keypoint(kp2_enum)
        if kp1 and kp2 and kp1.confidence >= min_conf and kp2.confidence >= min_conf:
            pt1 = kp1.to_pixel(w, h)
            pt2 = kp2.to_pixel(w, h)
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    for kp in pose.keypoints:
        if kp.confidence >= min_conf:
            pt = kp.to_pixel(w, h)
            cv2.circle(img, pt, 5, (0, 255, 0), -1)

    return img
