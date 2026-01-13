import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_and_classify
from nms import apply_nms_to_results
from config import MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE, DETECTION_THRESHOLD

# Load models
print("Loading models...")
face_detector = FaceDetector()
face_detector.load('models/face_detector.pkl')
char_recognizer = CharacterRecognizer()
char_recognizer.load('models/character_recognizer.pkl')

# Load image
img = cv2.imread('validare/validare/0005.jpg')
print(f'Image shape: {img.shape}')

# Ground truth for comparison
gt = {
    'fred': [(30, 90, 86, 156)],
    'daphne': [(140, 99, 193, 157)],
    'velma': [(259, 81, 363, 167)]
}

# Detect and classify
print(f'\nRunning detection with threshold={DETECTION_THRESHOLD}...')
results = detect_and_classify(
    img, face_detector, char_recognizer,
    min_window=MIN_WINDOW_SIZE,
    max_window=MAX_WINDOW_SIZE,
    scale_factor=SCALE_FACTOR,
    step_size=STEP_SIZE,
    threshold=DETECTION_THRESHOLD,
    show_progress=True
)

print(f'\nBefore NMS:')
print(f'  All faces: {len(results["all_faces"])}')
for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
    print(f'  {char_name}: {len(results["characters"][char_name])}')

# Apply NMS (modifies results in-place)
apply_nms_to_results(results)

print(f'\nAfter NMS:')
print(f'  All faces: {len(results["all_faces"])}')
for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
    print(f'  {char_name}: {len(results["characters"][char_name])}')

print(f'\n{"="*60}')
print(f'FINAL DETECTIONS')
print(f'{"="*60}')

print(f'\nAll faces ({len(results["all_faces"])}):')
for i, det in enumerate(results["all_faces"][:10]):
    print(f'  {i+1}. {det}')

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box1[:4]
    gx1, gy1, gx2, gy2 = box2
    
    xx1 = max(x1, gx1)
    yy1 = max(y1, gy1)
    xx2 = min(x2, gx2)
    yy2 = min(y2, gy2)
    
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_det = (x2 - x1) * (y2 - y1)
    area_gt = (gx2 - gx1) * (gy2 - gy1)
    union = area_det + area_gt - inter
    
    return inter / union if union > 0 else 0

for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
    dets = results["characters"][char_name]
    print(f'\n{char_name.upper()} ({len(dets)}):')
    
    if char_name in gt:
        print(f'  Ground Truth: {gt[char_name]}')
    
    for i, det in enumerate(dets):
        x1, y1, x2, y2, score = det
        print(f'  {i+1}. ({x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}) score={score:.3f}', end='')
        
        # Check IoU with GT
        if char_name in gt:
            for gt_box in gt[char_name]:
                iou = calculate_iou(det, gt_box)
                print(f' | IoU={iou:.3f}', end='')
        print()

print(f'\n{"="*60}')
print('Summary:')
print(f'{"="*60}')
for char_name in ['fred', 'daphne', 'velma']:
    if char_name in gt:
        dets = results["characters"][char_name]
        if len(dets) > 0:
            best_iou = max([calculate_iou(det, gt[char_name][0]) for det in dets])
            status = "✓" if best_iou >= 0.5 else "✗"
            print(f'{char_name.capitalize():8s}: {status} (best IoU: {best_iou:.3f})')
        else:
            print(f'{char_name.capitalize():8s}: ✗ (no detections)')
