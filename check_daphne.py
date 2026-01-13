import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_and_classify
from config import MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE, DETECTION_THRESHOLD

# Load models
face_detector = FaceDetector()
face_detector.load('models/face_detector.pkl')
char_recognizer = CharacterRecognizer()
char_recognizer.load('models/character_recognizer.pkl')

# Load image
img = cv2.imread('validare/validare/0005.jpg')

# Daphne ground truth
daphne_gt = (140, 99, 193, 157)
print(f'Daphne GT: {daphne_gt}')
print(f'Center: ({(daphne_gt[0] + daphne_gt[2])//2}, {(daphne_gt[1] + daphne_gt[3])//2})')
print(f'Size: {daphne_gt[2] - daphne_gt[0]} x {daphne_gt[3] - daphne_gt[1]}')

# Run detection
results = detect_and_classify(
    img, face_detector, char_recognizer,
    min_window=MIN_WINDOW_SIZE,
    max_window=MAX_WINDOW_SIZE,
    scale_factor=SCALE_FACTOR,
    step_size=STEP_SIZE,
    threshold=DETECTION_THRESHOLD,
    show_progress=False
)

print(f'\n=== ALL FACE DETECTIONS NEAR DAPHNE ===')
print('(within 50 pixels of Daphne GT top-left corner)')
for det in sorted(results['all_faces'], key=lambda x: -x[4])[:50]:
    x1, y1, x2, y2, score = det
    if abs(x1 - daphne_gt[0]) < 50 and abs(y1 - daphne_gt[1]) < 50:
        # Calculate IoU
        xx1 = max(x1, daphne_gt[0])
        yy1 = max(y1, daphne_gt[1])
        xx2 = min(x2, daphne_gt[2])
        yy2 = min(y2, daphne_gt[3])
        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        area_det = (x2 - x1) * (y2 - y1)
        area_gt = (daphne_gt[2] - daphne_gt[0]) * (daphne_gt[3] - daphne_gt[1])
        union = area_det + area_gt - inter
        iou = inter / union if union > 0 else 0
        print(f'  Face at ({x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}) score={score:.3f} IoU={iou:.3f}')

print(f'\n=== ALL DAPHNE CHARACTER DETECTIONS ===')
print('(sorted by score)')
for i, det in enumerate(sorted(results['characters']['daphne'], key=lambda x: -x[4])[:10]):
    x1, y1, x2, y2, score = det
    # Calculate IoU
    xx1 = max(x1, daphne_gt[0])
    yy1 = max(y1, daphne_gt[1])
    xx2 = min(x2, daphne_gt[2])
    yy2 = min(y2, daphne_gt[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_det = (x2 - x1) * (y2 - y1)
    area_gt = (daphne_gt[2] - daphne_gt[0]) * (daphne_gt[3] - daphne_gt[1])
    union = area_det + area_gt - inter
    iou = inter / union if union > 0 else 0
    print(f'  {i+1}. ({x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}) score={score:.3f} IoU={iou:.3f}')
