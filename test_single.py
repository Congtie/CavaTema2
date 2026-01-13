"""
Script rapid pentru testare pe o singură imagine.
"""

import os
import sys
import cv2

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from config import VALIDATION_IMAGES, MODELS_PATH, CHARACTERS, WINDOW_SIZE
from config import MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_and_classify
from nms import apply_nms_to_results


def test_single_image(image_name="0005.jpg"):
    """Test pe o singură imagine."""
    
    # Load models
    print("Loading models...")
    face_detector = FaceDetector()
    face_detector.load(os.path.join(MODELS_PATH, "face_detector.pkl"))
    
    char_recognizer = CharacterRecognizer()
    char_recognizer.load(os.path.join(MODELS_PATH, "character_recognizer.pkl"))
    
    # Load image
    img_path = os.path.join(VALIDATION_IMAGES, image_name)
    image = cv2.imread(img_path)
    print(f"Image: {image_name}, shape: {image.shape}")
    
    # Detect and classify
    print("Detecting...")
    from config import DETECTION_THRESHOLD
    results = detect_and_classify(
        image, face_detector, char_recognizer,
        min_window=MIN_WINDOW_SIZE,
        max_window=MAX_WINDOW_SIZE,
        scale_factor=SCALE_FACTOR,
        step_size=STEP_SIZE,
        threshold=DETECTION_THRESHOLD,
        show_progress=True
    )
    
    # Apply NMS - keep more detections per character for precision-recall curve
    from config import NMS_THRESHOLD
    results = apply_nms_to_results(results, NMS_THRESHOLD, max_detections_per_char=5)
    
    # Print results
    print(f"\nAll faces: {len(results['all_faces'])}")
    for char in CHARACTERS:
        print(f"{char}: {len(results['characters'][char])} detections")
        for det in results['characters'][char]:
            print(f"  -> {det}")
    
    # Ground truth for 0005.jpg (from validation)
    gt = {
        'fred': [(30, 90, 86, 156)],
        'daphne': [(140, 99, 193, 157)],
        'velma': [(259, 81, 363, 167)],
        'shaggy': []
    }
    
    # Visualize
    colors = {
        'fred': (0, 165, 255),    # Orange
        'daphne': (128, 0, 128),  # Purple
        'shaggy': (0, 128, 0),    # Green
        'velma': (0, 0, 255),     # Red
    }
    
    vis_image = image.copy()
    
    # Draw ground truth first (dashed/thinner lines)
    for char in CHARACTERS:
        color = colors[char]
        for xmin, ymin, xmax, ymax in gt.get(char, []):
            # Draw GT in lighter color with thinner line
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(vis_image, f"GT:{char}", (xmin, ymax + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw predictions
    for char in CHARACTERS:
        color = colors[char]
        for xmin, ymin, xmax, ymax, score in results['characters'][char]:
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{char}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (xmin, ymin - th - 5), (xmin + tw, ymin), color, -1)
            cv2.putText(vis_image, label, (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend
    y = 20
    for char, color in colors.items():
        cv2.rectangle(vis_image, (10, y - 12), (25, y), color, -1)
        cv2.putText(vis_image, char, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 20
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "vizualizari", f"test_{image_name}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"\nSaved: {output_path}")
    
    return results


if __name__ == "__main__":
    test_single_image("0005.jpg")
