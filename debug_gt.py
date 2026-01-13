import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from config import *
from feature_extraction import extract_hog_features
from classifier import FaceDetector

# Load
fd = FaceDetector()
fd.load('models/face_detector.pkl')
img = cv2.imread('validare/validare/0100.jpg')
h, w = img.shape[:2]

# GT is (167, 126, 334, 297) - center is around (250, 211)
gt = (167, 126, 334, 297)
gt_center_x = (gt[0] + gt[2]) // 2  # 250
gt_center_y = (gt[1] + gt[3]) // 2  # 211

print(f"GT: {gt}")
print(f"GT center: ({gt_center_x}, {gt_center_y})")
print(f"GT size: {gt[2]-gt[0]}x{gt[3]-gt[1]}")

# Try scale for 187px window
window_size = 187
scale = 64 / window_size
print(f"\nUsing scale for {window_size}px window: {scale:.3f}")

# Resize image
new_w = int(w * scale)
new_h = int(h * scale)
scaled = cv2.resize(img, (new_w, new_h))
print(f"Scaled image: {new_w}x{new_h}")

# Find GT center in scaled image
scaled_cx = int(gt_center_x * scale)
scaled_cy = int(gt_center_y * scale)
print(f"GT center in scaled: ({scaled_cx}, {scaled_cy})")

# Extract 64x64 window at GT center
win_h, win_w = WINDOW_SIZE
x1 = scaled_cx - win_w // 2
y1 = scaled_cy - win_h // 2
x2 = x1 + win_w
y2 = y1 + win_h

print(f"Window at: ({x1}, {y1}, {x2}, {y2})")

# Clip to image bounds
x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(new_w, x2)
y2 = min(new_h, y2)

if x2 - x1 == win_w and y2 - y1 == win_h:
    window = scaled[y1:y2, x1:x2]
    print(f"Window shape: {window.shape}")
    
    # Extract features
    features = extract_hog_features(window)
    print(f"Features shape: {features.shape}")
    
    # Get score
    features_scaled = fd.scaler.transform([features])
    score = fd.classifier.decision_function(features_scaled)[0]
    print(f"\nFace detector score at GT location: {score:.3f}")
    print(f"Threshold: {DETECTION_THRESHOLD}")
    if score >= DETECTION_THRESHOLD:
        print("PASS - Would be detected")
    else:
        print("FAIL - Below threshold")
else:
    print("Window clipped - can't test")
