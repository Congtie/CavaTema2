import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_and_classify
from config import MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE, DETECTION_THRESHOLD, NMS_THRESHOLD

# Load models
face_detector = FaceDetector()
face_detector.load('models/face_detector.pkl')
char_recognizer = CharacterRecognizer()
char_recognizer.load('models/character_recognizer.pkl')

# Load image
img = cv2.imread('validare/validare/0005.jpg')

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

# Daphne detections with good IoU
daphne_good_dets = [
    (146, 101, 191, 146, 6.494),
    (146, 106, 191, 151, 6.420),
]

print(f'NMS_THRESHOLD = {NMS_THRESHOLD}')
print(f'\nChecking what suppresses the good Daphne detections...\n')

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

for good_det in daphne_good_dets:
    print(f'Good Daphne detection: {good_det[:4]} score={good_det[4]:.3f}')
    
    # Check against all character detections with higher score
    suppressed_by = []
    for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
        for det in results['characters'][char_name]:
            if det[4] > good_det[4]:  # Higher score
                iou = compute_iou(good_det, det)
                if iou > NMS_THRESHOLD:
                    suppressed_by.append((char_name, det, iou))
    
    if suppressed_by:
        print(f'  SUPPRESSED BY:')
        for char_name, det, iou in sorted(suppressed_by, key=lambda x: -x[1][4]):
            print(f'    {char_name}: {det[:4]} score={det[4]:.3f} IoU={iou:.3f}')
    else:
        print(f'  NOT suppressed - should be kept!')
    print()
