#!/usr/bin/env python3
"""
Quick System Test
Run this first to verify everything works
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_all():
    print("=" * 50)
    print("SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: NPU
    print("\n[1] NPU Device...")
    if os.path.exists('/dev/ethosu0'):
        print("  ✓ NPU found")
    else:
        print("  ✗ NPU NOT found")
    
    # Test 2: Delegate
    print("\n[2] NPU Delegate...")
    if os.path.exists('/usr/lib/libethosu_delegate.so'):
        print("  ✓ Delegate found")
    else:
        print("  ✗ Delegate NOT found")
    
    # Test 3: Model
    print("\n[3] Model File...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'movenet_lightning_int8_vela.tflite')
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024
        print(f"  ✓ Model found ({size:.0f} KB)")
    else:
        print(f"  ✗ Model NOT found at {model_path}")
        return False
    
    # Test 4: Camera
    print("\n[4] Camera...")
    if os.path.exists('/dev/video0'):
        print("  ✓ Camera found")
    else:
        print("  ✗ Camera NOT found - plug in webcam")
    
    # Test 5: TFLite
    print("\n[5] TFLite Runtime...")
    try:
        import tflite_runtime.interpreter as tflite
        print("  ✓ tflite_runtime OK")
    except:
        try:
            import tensorflow.lite as tflite
            print("  ✓ tensorflow.lite OK")
        except:
            print("  ✗ TFLite NOT found")
            return False
    
    # Test 6: OpenCV
    print("\n[6] OpenCV...")
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except:
        print("  ✗ OpenCV NOT found")
        return False
    
    # Test 7: Quick inference
    print("\n[7] NPU Inference Test...")
    try:
        import numpy as np
        delegate = tflite.load_delegate('/usr/lib/libethosu_delegate.so')
        interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
        interpreter.allocate_tensors()
        
        inp = interpreter.get_input_details()[0]
        test_input = np.random.randint(0, 255, inp['shape'], dtype=np.uint8)
        
        import time
        interpreter.set_tensor(inp['index'], test_input)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        
        print(f"  ✓ NPU inference: {(t1-t0)*1000:.1f}ms")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nRun the app with:")
    print("  cd /home/root/elderly_fall_detection")
    print("  python3 app/main.py")
    print("")
    return True


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
