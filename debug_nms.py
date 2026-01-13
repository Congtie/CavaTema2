"""Debug NMS results."""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from config import MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_and_classify
from nms import apply_nms_to_results, compute_iou

# Load
face_detector = FaceDetector()
face_detector.load('models/face_detector.pkl')
char_recognizer = CharacterRecognizer()
char_recognizer.load('models/character_recognizer.pkl')

img = cv2.imread('validare/validare/0005.jpg')

# Detect with high threshold
results = detect_and_classify(
    img, face_detector, char_recognizer,
    min_window=MIN_WINDOW_SIZE,
    max_window=MAX_WINDOW_SIZE,
    scale_factor=SCALE_FACTOR,
    step_size=STEP_SIZE,
    threshold=4.0,  # high threshold
    show_progress=False
)

# Apply NMS
results = apply_nms_to_results(results, 0.3, max_detections_per_char=2)

# GT
gt = {
    'fred': (30, 90, 86, 156),
    'daphne': (140, 99, 193, 157),
    'velma': (259, 81, 363, 167),
}

print(f'All faces after NMS: {len(results["all_faces"])}')
print()
print('Characters after NMS:')
for char in ['fred', 'daphne', 'shaggy', 'velma']:
    dets = results['characters'][char]
    print(f'{char}: {len(dets)} detections')
    for det in dets:
        # Check IoU with GT
        best_iou = 0
        best_gt = None
        for gt_char, gt_box in gt.items():
            iou = compute_iou(det[:4], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_char
        status = f'IoU={best_iou:.2f} with {best_gt}' if best_iou > 0.1 else 'FALSE POSITIVE'
        print(f'  {det[:4]} score={det[4]:.2f} -> {status}')
