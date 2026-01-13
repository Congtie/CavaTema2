"""
Sliding Window Detection module.
Implements multi-scale sliding window approach for face detection.
"""

import numpy as np
import cv2
from typing import List, Tuple, Generator
from tqdm import tqdm

from config import (
    WINDOW_SIZE, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE,
    SCALE_FACTOR, STEP_SIZE, DETECTION_THRESHOLD
)
from feature_extraction import extract_hog_features, extract_combined_features


def sliding_window(
    image: np.ndarray,
    step_size: int = STEP_SIZE,
    window_size: Tuple[int, int] = WINDOW_SIZE
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Generate sliding window positions across an image.
    
    Args:
        image: Input image
        step_size: Step size in pixels
        window_size: (height, width) of the window
        
    Yields:
        (x, y, window) for each position
    """
    h, w = image.shape[:2]
    win_h, win_w = window_size
    
    for y in range(0, h - win_h + 1, step_size):
        for x in range(0, w - win_w + 1, step_size):
            window = image[y:y + win_h, x:x + win_w]
            yield (x, y, window)


def image_pyramid(
    image: np.ndarray,
    scale_factor: float = SCALE_FACTOR,
    min_size: Tuple[int, int] = WINDOW_SIZE
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Generate an image pyramid by progressively scaling down.
    
    Args:
        image: Input image
        scale_factor: Scale factor between levels
        min_size: Minimum image size
        
    Yields:
        (scaled_image, scale) for each level
    """
    yield (image, 1.0)
    
    current_scale = 1.0
    
    while True:
        current_scale /= scale_factor
        
        h, w = image.shape[:2]
        new_h = int(h * current_scale)
        new_w = int(w * current_scale)
        
        if new_h < min_size[0] or new_w < min_size[1]:
            break
        
        scaled_image = cv2.resize(image, (new_w, new_h))
        yield (scaled_image, current_scale)


def detect_faces_single_scale(
    image: np.ndarray,
    classifier,
    step_size: int = STEP_SIZE,
    threshold: float = DETECTION_THRESHOLD,
    use_color: bool = False
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces in an image at a single scale.
    
    Args:
        image: Input image
        classifier: Trained face detector
        step_size: Step size for sliding window
        threshold: Detection score threshold
        use_color: Whether classifier uses color features
        
    Returns:
        List of (xmin, ymin, xmax, ymax, score) detections
    """
    detections = []
    h, w = image.shape[:2]
    win_h, win_w = WINDOW_SIZE
    
    # Collect all windows first for batch processing
    windows = []
    positions = []
    
    for x, y, window in sliding_window(image, step_size, WINDOW_SIZE):
        windows.append(window)
        positions.append((x, y))
    
    if not windows:
        return detections
    
    # Extract features and predict in batch
    if use_color:
        from feature_extraction import extract_combined_features_batch
        features = extract_combined_features_batch(windows, use_color=True, show_progress=False)
    else:
        from feature_extraction import extract_hog_features_batch
        features = extract_hog_features_batch(windows, show_progress=False)
    
    # Get predictions
    features_scaled = classifier.scaler.transform(features)
    scores = classifier.classifier.decision_function(features_scaled)
    
    # Filter by threshold
    for i, (x, y) in enumerate(positions):
        if scores[i] > threshold:
            xmin, ymin = x, y
            xmax, ymax = x + win_w, y + win_h
            detections.append((xmin, ymin, xmax, ymax, scores[i]))
    
    return detections


def detect_faces_multiscale(
    image: np.ndarray,
    classifier,
    min_window: int = MIN_WINDOW_SIZE,
    max_window: int = MAX_WINDOW_SIZE,
    scale_factor: float = SCALE_FACTOR,
    step_size: int = STEP_SIZE,
    threshold: float = DETECTION_THRESHOLD,
    use_color: bool = False,
    show_progress: bool = True
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces in an image using multi-scale sliding window.
    
    Args:
        image: Input image
        classifier: Trained face detector
        min_window: Minimum detection window size
        max_window: Maximum detection window size
        scale_factor: Scale factor for image pyramid
        step_size: Step size for sliding window
        threshold: Detection score threshold
        use_color: Whether classifier uses color features
        show_progress: Whether to show progress bar
        
    Returns:
        List of (xmin, ymin, xmax, ymax, score) detections in original image coordinates
    """
    all_detections = []
    h, w = image.shape[:2]
    win_h, win_w = WINDOW_SIZE
    
    # Calculate scales to cover the range of window sizes
    # If we want to detect faces of size S, we need to scale the image by WINDOW_SIZE/S
    scales = []
    current_window = min_window
    while current_window <= max_window:
        scale = win_w / current_window
        if scale * w >= win_w and scale * h >= win_h:
            scales.append(scale)
        current_window = int(current_window * scale_factor)
    
    iterator = tqdm(scales, desc="Multi-scale detection") if show_progress else scales
    
    for scale in iterator:
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w < win_w or new_h < win_h:
            continue
        
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # Detect at this scale
        detections = detect_faces_single_scale(
            scaled_image, classifier, step_size, threshold, use_color
        )
        
        # Convert detections back to original image coordinates
        for xmin, ymin, xmax, ymax, score in detections:
            orig_xmin = int(xmin / scale)
            orig_ymin = int(ymin / scale)
            orig_xmax = int(xmax / scale)
            orig_ymax = int(ymax / scale)
            all_detections.append((orig_xmin, orig_ymin, orig_xmax, orig_ymax, score))
    
    return all_detections


def detect_and_classify(
    image: np.ndarray,
    face_detector,
    character_classifier,
    min_window: int = MIN_WINDOW_SIZE,
    max_window: int = MAX_WINDOW_SIZE,
    scale_factor: float = SCALE_FACTOR,
    step_size: int = STEP_SIZE,
    threshold: float = DETECTION_THRESHOLD,
    show_progress: bool = True
) -> dict:
    """
    Detect faces and classify characters.
    
    Args:
        image: Input image
        face_detector: Trained face detector (FaceDetector)
        character_classifier: Trained character recognizer (CharacterRecognizer)
        min_window: Minimum detection window size
        max_window: Maximum detection window size
        scale_factor: Scale factor for image pyramid
        step_size: Step size for sliding window
        threshold: Detection score threshold
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with:
            - 'all_faces': List of (xmin, ymin, xmax, ymax, score) for Task 1
            - 'characters': Dict mapping character name to list of (xmin, ymin, xmax, ymax, score)
    """
    from config import CHARACTERS, CHAR_DETECTION_THRESHOLDS
    
    # First, detect all faces with a lower threshold to catch more candidates
    min_threshold = min(CHAR_DETECTION_THRESHOLDS.values())
    face_detections = detect_faces_multiscale(
        image, face_detector,
        min_window=min_window,
        max_window=max_window,
        scale_factor=scale_factor,
        step_size=step_size,
        threshold=min_threshold,  # Use minimum threshold to catch all potential faces
        use_color=face_detector.use_color,
        show_progress=show_progress
    )
    
    # Initialize results
    results = {
        'all_faces': face_detections,
        'characters': {char: [] for char in CHARACTERS}
    }
    
    if not face_detections:
        return results
    
    # Extract patches for classification
    patches = []
    for xmin, ymin, xmax, ymax, score in face_detections:
        # Ensure valid coordinates
        h, w = image.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        
        patch = image[ymin:ymax, xmin:xmax]
        if patch.size > 0:
            patch = cv2.resize(patch, (WINDOW_SIZE[1], WINDOW_SIZE[0]))
            patches.append(patch)
        else:
            patches.append(np.zeros((*WINDOW_SIZE, 3), dtype=np.uint8))
    
    # Classify all patches
    char_predictions = character_classifier.predict(patches)
    char_scores = character_classifier.predict_scores(patches)
    
    # Minimum character score threshold - lower to keep more detections
    CHAR_SCORE_THRESHOLD = 0.5
    
    # Organize by character with per-character thresholds
    for i, (xmin, ymin, xmax, ymax, face_score) in enumerate(face_detections):
        char_idx = char_predictions[i]
        char_name = CHARACTERS[char_idx]
        
        # Apply per-character detection threshold
        char_threshold = CHAR_DETECTION_THRESHOLDS.get(char_name, threshold)
        if face_score < char_threshold:
            continue  # Skip detections below character-specific threshold
        
        # Use character-specific score for better ranking
        if len(char_scores.shape) > 1:
            char_score = char_scores[i, char_idx]
        else:
            char_score = char_scores[i]
        
        # Only keep detections with high character confidence
        if char_score >= CHAR_SCORE_THRESHOLD:
            # Use combined score: face_score + char_score for better ranking
            combined_score = face_score + char_score
            results['characters'][char_name].append((xmin, ymin, xmax, ymax, float(combined_score)))
    
    return results


if __name__ == "__main__":
    # Test sliding window on a sample image
    import os
    from config import VALIDATION_IMAGES
    
    # Load a sample image
    img_path = os.path.join(VALIDATION_IMAGES, "0001.jpg")
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        print(f"Image shape: {image.shape}")
        
        # Test sliding window
        count = 0
        for x, y, window in sliding_window(image, step_size=16, window_size=WINDOW_SIZE):
            count += 1
        print(f"Number of windows at original scale: {count}")
        
        # Test image pyramid
        for scaled_img, scale in image_pyramid(image, scale_factor=1.2):
            print(f"Scale: {scale:.3f}, Size: {scaled_img.shape[:2]}")
    else:
        print("Sample image not found")
