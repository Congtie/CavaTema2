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
img = cv2.imread('validare/validare/0100.jpg')
print(f'Image shape: {img.shape}')

# Ground truth for 0100.jpg - only Daphne
gt = {
    'daphne': [(167, 126, 334, 297)]
}

# Run detection
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

# Apply NMS
apply_nms_to_results(results)

print(f'\nAfter NMS:')
print(f'  All faces: {len(results["all_faces"])}')
for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
    print(f'  {char_name}: {len(results["characters"][char_name])}')

def calculate_iou(box1, box2):
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

# Create visualization
vis_img = img.copy()
colors = {
    'fred': (255, 0, 0),
    'daphne': (255, 0, 255),
    'shaggy': (0, 255, 0),
    'velma': (0, 165, 255),
    'gt': (0, 255, 255)
}

# Draw ground truth
print("\n" + "="*60)
print("Ground Truth:")
for char_name, boxes in gt.items():
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors['gt'], 2)
        cv2.putText(vis_img, f'GT-{char_name.upper()}', (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['gt'], 2)
        print(f"  {char_name}: ({x1}, {y1}, {x2}, {y2})")

print("\nDetections:")
for char_name in ['fred', 'daphne', 'shaggy', 'velma']:
    dets = results['characters'][char_name]
    if len(dets) > 0:
        print(f"\n{char_name.upper()} ({len(dets)}):")
        for i, det in enumerate(dets):
            x1, y1, x2, y2, score = det
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors[char_name], 2)
            cv2.putText(vis_img, f'{char_name[0].upper()}:{score:.1f}', 
                        (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[char_name], 1)
            
            if char_name in gt:
                for gt_box in gt[char_name]:
                    iou = calculate_iou(det, gt_box)
                    print(f"  {i+1}. ({x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}) score={score:.2f} IoU={iou:.3f}")
            else:
                print(f"  {i+1}. ({x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}) score={score:.2f}")

print("\n" + "="*60)
print("Summary:")
for char_name in gt.keys():
    dets = results['characters'][char_name]
    if len(dets) > 0:
        best_iou = max([calculate_iou(det, gt[char_name][0]) for det in dets])
        status = "✓" if best_iou >= 0.5 else "✗"
        print(f'{char_name.capitalize():8s}: {status} (best IoU: {best_iou:.3f})')
    else:
        print(f'{char_name.capitalize():8s}: ✗ (no detections)')

cv2.imwrite('detection_0100.jpg', vis_img)
print(f"\nVisualization saved to: detection_0100.jpg")

cv2.imshow('Detections on 0100.jpg', vis_img)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
