"""
Script to improve detection by:
1. Data augmentation for underrepresented classes (Shaggy)
2. Per-character detection thresholds
3. Hard negative mining
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    TRAIN_PATH, TRAIN_ANNOTATIONS, TRAIN_IMAGES,
    WINDOW_SIZE, CHARACTERS, CHAR_TO_LABEL, MODELS_PATH
)
from data_loader import (
    load_all_annotations, extract_patch, extract_positive_patches,
    extract_negative_patches, compute_iou
)
from classifier import FaceDetector, CharacterRecognizer


def augment_patch(patch, augmentations=['flip', 'brightness', 'contrast']):
    """
    Apply data augmentation to a patch.
    
    Args:
        patch: Input image patch
        augmentations: List of augmentations to apply
        
    Returns:
        List of augmented patches (including original)
    """
    augmented = [patch]
    
    if 'flip' in augmentations:
        # Horizontal flip
        flipped = cv2.flip(patch, 1)
        augmented.append(flipped)
    
    if 'brightness' in augmentations:
        # Brightness variations
        for beta in [-30, 30]:
            bright = cv2.convertScaleAbs(patch, alpha=1.0, beta=beta)
            augmented.append(bright)
    
    if 'contrast' in augmentations:
        # Contrast variations
        for alpha in [0.8, 1.2]:
            contrast = cv2.convertScaleAbs(patch, alpha=alpha, beta=0)
            augmented.append(contrast)
    
    if 'rotation' in augmentations:
        # Small rotations
        h, w = patch.shape[:2]
        center = (w // 2, h // 2)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(patch, M, (w, h))
            augmented.append(rotated)
    
    return augmented


def prepare_augmented_training_data():
    """
    Prepare training data with augmentation for underrepresented classes.
    Balances the dataset by augmenting Shaggy and Fred more.
    """
    all_annotations = load_all_annotations()
    
    # Count samples per character
    counts = {}
    for char in CHARACTERS:
        if char in all_annotations:
            total = sum(len(boxes) for boxes in all_annotations[char].values())
            counts[char] = total
    
    print("Original sample counts:")
    for char, count in counts.items():
        print(f"  {char}: {count}")
    
    max_count = max(counts.values())
    
    all_positive_patches = []
    all_positive_labels = []
    
    # Extract and augment patches
    for character in CHARACTERS:
        if character not in all_annotations:
            continue
            
        image_folder = TRAIN_IMAGES[character]
        annotations = all_annotations[character]
        
        # Extract original patches
        patches, labels = extract_positive_patches(image_folder, annotations)
        
        # Calculate augmentation factor to balance classes
        current_count = len(patches)
        target_count = max_count
        aug_factor = target_count / current_count if current_count > 0 else 1
        
        print(f"\n{character}: {current_count} original, target: {target_count}, aug_factor: {aug_factor:.2f}")
        
        if aug_factor > 1.5:
            # Need heavy augmentation (Shaggy, Fred)
            print(f"  Applying full augmentation...")
            for patch, label in zip(patches, labels):
                augmented = augment_patch(patch, ['flip', 'brightness', 'contrast', 'rotation'])
                # Keep only enough to reach target
                keep_count = min(len(augmented), int(aug_factor) + 1)
                for aug_patch in augmented[:keep_count]:
                    all_positive_patches.append(aug_patch)
                    all_positive_labels.append(label)
        elif aug_factor > 1.2:
            # Moderate augmentation
            print(f"  Applying moderate augmentation...")
            for patch, label in zip(patches, labels):
                augmented = augment_patch(patch, ['flip', 'brightness'])
                keep_count = min(len(augmented), int(aug_factor) + 1)
                for aug_patch in augmented[:keep_count]:
                    all_positive_patches.append(aug_patch)
                    all_positive_labels.append(label)
        else:
            # No augmentation needed (Daphne)
            print(f"  No augmentation needed")
            all_positive_patches.extend(patches)
            all_positive_labels.extend(labels)
    
    print(f"\nTotal positive patches after augmentation: {len(all_positive_patches)}")
    
    # Count augmented samples per character
    aug_counts = {}
    for label in all_positive_labels:
        char = [k for k, v in CHAR_TO_LABEL.items() if v == label][0]
        aug_counts[char] = aug_counts.get(char, 0) + 1
    
    print("\nAugmented sample counts:")
    for char, count in aug_counts.items():
        print(f"  {char}: {count}")
    
    # Extract negative patches (use all annotations to avoid faces)
    all_negative_patches = []
    for character in CHARACTERS:
        if character not in all_annotations:
            continue
        image_folder = TRAIN_IMAGES[character]
        neg_patches = extract_negative_patches(
            image_folder, 
            all_annotations[character],
            all_annotations=all_annotations
        )
        all_negative_patches.extend(neg_patches)
    
    print(f"Total negative patches: {len(all_negative_patches)}")
    
    # Combine
    X = np.array(all_positive_patches + all_negative_patches)
    y = np.array(all_positive_labels + [CHAR_TO_LABEL['negative']] * len(all_negative_patches))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y, np.array(all_positive_patches), np.array(all_positive_labels)


def train_with_augmentation():
    """Train models with augmented data."""
    print("=" * 60)
    print("TRAINING WITH DATA AUGMENTATION")
    print("=" * 60)
    
    X, y, X_pos, y_pos = prepare_augmented_training_data()
    
    # Train face detector
    print("\n" + "=" * 60)
    print("Training Face Detector (Task 1)")
    print("=" * 60)
    
    y_binary = (y != CHAR_TO_LABEL['negative']).astype(int)
    
    face_detector = FaceDetector(use_color=False)
    face_detector.train(X, y_binary, validate=True)
    face_detector.save("face_detector_augmented.pkl")
    
    # Train character recognizer
    print("\n" + "=" * 60)
    print("Training Character Recognizer (Task 2)")
    print("=" * 60)
    
    char_recognizer = CharacterRecognizer(use_color=True)
    char_recognizer.train(X_pos, y_pos, validate=True)
    char_recognizer.save("character_recognizer_augmented.pkl")
    
    return face_detector, char_recognizer


def hard_negative_mining(face_detector, X_neg_original, max_hard_negatives=5000):
    """
    Find hard negatives (false positives with high scores).
    
    Args:
        face_detector: Trained face detector
        X_neg_original: Original negative samples
        max_hard_negatives: Maximum hard negatives to collect
        
    Returns:
        Array of hard negative patches
    """
    print("\n" + "=" * 60)
    print("HARD NEGATIVE MINING")
    print("=" * 60)
    
    # Get scores for all negative samples
    print("Scoring negative samples...")
    scores = face_detector.predict_scores(list(X_neg_original))
    
    # Find hard negatives (high scores = false positives)
    hard_indices = np.where(scores > 0)[0]  # Misclassified as faces
    print(f"Found {len(hard_indices)} hard negatives (score > 0)")
    
    # Sort by score and keep top ones
    sorted_indices = hard_indices[np.argsort(scores[hard_indices])[::-1]]
    top_indices = sorted_indices[:max_hard_negatives]
    
    print(f"Keeping top {len(top_indices)} hard negatives")
    
    return X_neg_original[top_indices]


def train_with_hard_negatives():
    """Train with hard negative mining."""
    print("=" * 60)
    print("PHASE 1: Initial training with augmentation")
    print("=" * 60)
    
    X, y, X_pos, y_pos = prepare_augmented_training_data()
    
    # Separate negative samples
    neg_mask = (y == CHAR_TO_LABEL['negative'])
    X_neg = X[neg_mask]
    
    # Train initial face detector
    y_binary = (y != CHAR_TO_LABEL['negative']).astype(int)
    face_detector = FaceDetector(use_color=False)
    face_detector.train(X, y_binary, validate=True)
    
    # Find hard negatives
    hard_negatives = hard_negative_mining(face_detector, X_neg)
    
    if len(hard_negatives) > 0:
        print("\n" + "=" * 60)
        print("PHASE 2: Retraining with hard negatives")
        print("=" * 60)
        
        # Add hard negatives to training data (with extra weight by duplicating)
        X_with_hard = np.concatenate([X, hard_negatives, hard_negatives])  # Add 2x
        y_with_hard = np.concatenate([
            y, 
            np.full(len(hard_negatives), CHAR_TO_LABEL['negative']),
            np.full(len(hard_negatives), CHAR_TO_LABEL['negative'])
        ])
        
        y_binary_with_hard = (y_with_hard != CHAR_TO_LABEL['negative']).astype(int)
        
        print(f"Training data size: {len(X_with_hard)} (including {2*len(hard_negatives)} hard negatives)")
        
        # Retrain
        face_detector = FaceDetector(use_color=False)
        face_detector.train(X_with_hard, y_binary_with_hard, validate=True)
    
    face_detector.save("face_detector.pkl")
    
    # Train character recognizer with augmented positives
    print("\n" + "=" * 60)
    print("Training Character Recognizer (Task 2)")
    print("=" * 60)
    
    char_recognizer = CharacterRecognizer(use_color=True)
    char_recognizer.train(X_pos, y_pos, validate=True)
    char_recognizer.save("character_recognizer.pkl")
    
    return face_detector, char_recognizer


if __name__ == "__main__":
    train_with_hard_negatives()
    print("\nTraining complete! Models saved to:", MODELS_PATH)
