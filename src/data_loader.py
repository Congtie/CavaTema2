"""
Data loading and preprocessing utilities.
Handles loading images, annotations, and extracting positive/negative patches.
"""

import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import (
    TRAIN_PATH, TRAIN_ANNOTATIONS, TRAIN_IMAGES,
    WINDOW_SIZE, MIN_POSITIVE_IOU, MAX_NEGATIVE_IOU,
    NUM_NEGATIVE_SAMPLES_PER_IMAGE, CHARACTERS, CHAR_TO_LABEL
)


def load_annotations(annotation_file: str) -> Dict[str, List[Tuple]]:
    """
    Load annotations from a file.
    
    Args:
        annotation_file: Path to the annotation file
        
    Returns:
        Dictionary mapping image names to list of (xmin, ymin, xmax, ymax, character) tuples
    """
    annotations = {}
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
                
            img_name = parts[0]
            xmin, ymin, xmax, ymax = map(int, parts[1:5])
            character = parts[5]
            
            if img_name not in annotations:
                annotations[img_name] = []
            annotations[img_name].append((xmin, ymin, xmax, ymax, character))
    
    return annotations


def load_all_annotations() -> Dict[str, Dict[str, List[Tuple]]]:
    """
    Load all annotations from all character folders.
    
    Returns:
        Dictionary mapping folder names to annotation dictionaries
    """
    all_annotations = {}
    
    for character, ann_file in TRAIN_ANNOTATIONS.items():
        if os.path.exists(ann_file):
            all_annotations[character] = load_annotations(ann_file)
            print(f"Loaded {len(all_annotations[character])} images annotations for {character}")
    
    return all_annotations


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: (xmin, ymin, xmax, ymax)
        box2: (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def extract_patch(image: np.ndarray, box: Tuple, target_size: Tuple = WINDOW_SIZE, padding: float = 0.2) -> Optional[np.ndarray]:
    """
    Extract and resize a patch from an image with optional padding.
    
    Args:
        image: Input image
        box: (xmin, ymin, xmax, ymax)
        target_size: Target size for the patch (height, width)
        padding: Fraction to expand the box (0.2 = 20% expansion on each side)
        
    Returns:
        Resized patch or None if invalid
    """
    xmin, ymin, xmax, ymax = box[:4]
    
    # Calculate padding
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    # Expand the box
    xmin -= pad_x
    ymin -= pad_y
    xmax += pad_x
    ymax += pad_y
    
    # Ensure valid coordinates
    h, w = image.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    patch = image[ymin:ymax, xmin:xmax]
    
    if patch.size == 0:
        return None
    
    # Resize to target size
    patch = cv2.resize(patch, (target_size[1], target_size[0]))
    
    return patch


def augment_patch(patch: np.ndarray) -> List[np.ndarray]:
    """
    Apply data augmentation to a face patch.

    Args:
        patch: Input patch

    Returns:
        List of augmented patches (includes original + augmented versions)
    """
    augmented = [patch]  # Original

    # Horizontal flip (mirror)
    flipped = cv2.flip(patch, 1)
    augmented.append(flipped)

    # Small rotations (±5 degrees)
    h, w = patch.shape[:2]
    center = (w // 2, h // 2)

    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    return augmented


def extract_positive_patches(
    image_folder: str,
    annotations: Dict[str, List[Tuple]],
    target_size: Tuple = WINDOW_SIZE,
    apply_augmentation: bool = False
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract positive patches (faces) from images with optional data augmentation.

    Args:
        image_folder: Path to folder containing images
        annotations: Dictionary of annotations
        target_size: Target patch size
        apply_augmentation: If True, apply data augmentation (flip + rotations)

    Returns:
        List of patches and their corresponding labels
    """
    patches = []
    labels = []

    for img_name, face_boxes in tqdm(annotations.items(), desc="Extracting positive patches"):
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        for box in face_boxes:
            patch = extract_patch(image, box, target_size)
            if patch is not None:
                character = box[4] if len(box) > 4 else "unknown"
                label = CHAR_TO_LABEL.get(character, CHAR_TO_LABEL["unknown"])

                if apply_augmentation:
                    # Apply augmentation: original + flipped + 2 rotations = 4x data
                    augmented_patches = augment_patch(patch)
                    for aug_patch in augmented_patches:
                        patches.append(aug_patch)
                        labels.append(label)
                else:
                    patches.append(patch)
                    labels.append(label)

    return patches, labels


def extract_negative_patches(
    image_folder: str,
    annotations: Dict[str, List[Tuple]],
    num_per_image: int = NUM_NEGATIVE_SAMPLES_PER_IMAGE,
    target_size: Tuple = WINDOW_SIZE,
    max_iou: float = MAX_NEGATIVE_IOU,
    all_annotations: Dict[str, Dict[str, List[Tuple]]] = None
) -> List[np.ndarray]:
    """
    Extract negative patches (non-faces) from images.
    
    Args:
        image_folder: Path to folder containing images
        annotations: Dictionary of annotations for this folder
        num_per_image: Number of negative patches per image
        target_size: Target patch size
        max_iou: Maximum allowed IoU with any face
        all_annotations: Dictionary of ALL annotations from ALL characters (to avoid faces from other characters)
        
    Returns:
        List of negative patches
    """
    patches = []
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    for img_name in tqdm(image_files, desc="Extracting negative patches"):
        img_path = os.path.join(image_folder, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Collect ALL face boxes from ALL characters for this image
        all_face_boxes = []
        if all_annotations is not None:
            for char_name, char_annotations in all_annotations.items():
                if img_name in char_annotations:
                    all_face_boxes.extend(char_annotations[img_name])
        else:
            # Fallback to just this folder's annotations
            all_face_boxes = annotations.get(img_name, [])
        
        attempts = 0
        max_attempts = num_per_image * 10
        collected = 0
        
        while collected < num_per_image and attempts < max_attempts:
            attempts += 1
            
            # Random patch size (aspect ratio close to 1:1)
            patch_size = np.random.randint(target_size[0], min(h, w) // 2 + 1)
            
            # Random position
            xmin = np.random.randint(0, max(1, w - patch_size))
            ymin = np.random.randint(0, max(1, h - patch_size))
            xmax = xmin + patch_size
            ymax = ymin + patch_size
            
            # Check if this patch overlaps too much with ANY face from ANY character
            is_negative = True
            for face_box in all_face_boxes:
                iou = compute_iou((xmin, ymin, xmax, ymax), face_box[:4])
                if iou > max_iou:
                    is_negative = False
                    break
            
            if is_negative:
                # NO padding for negative samples - just resize the random region as-is
                patch = extract_patch(image, (xmin, ymin, xmax, ymax), target_size, padding=0.0)
                if patch is not None:
                    patches.append(patch)
                    collected += 1
    
    return patches


def prepare_training_data(
    include_character_labels: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare complete training data with positive and negative samples.
    
    Args:
        include_character_labels: If True, return character-specific labels (Task 2)
                                 If False, return binary face/non-face labels (Task 1)
    
    Returns:
        X: Array of patches
        y: Array of labels
    """
    all_positive_patches = []
    all_positive_labels = []
    all_negative_patches = []
    
    # Load all annotations
    all_annotations = load_all_annotations()
    
    # Extract positive patches from all character folders
    for character in CHARACTERS:
        if character not in all_annotations:
            continue
            
        image_folder = TRAIN_IMAGES[character]
        annotations = all_annotations[character]
        
        patches, labels = extract_positive_patches(
            image_folder, annotations, WINDOW_SIZE
        )
        
        all_positive_patches.extend(patches)
        all_positive_labels.extend(labels)
        print(f"Extracted {len(patches)} positive patches from {character}")
    
    # Extract negative patches from all character folders
    # Pass ALL annotations to avoid selecting faces from other characters as negatives
    for character in CHARACTERS:
        if character not in all_annotations:
            continue
            
        image_folder = TRAIN_IMAGES[character]
        annotations = all_annotations[character]
        
        neg_patches = extract_negative_patches(
            image_folder, annotations, NUM_NEGATIVE_SAMPLES_PER_IMAGE, WINDOW_SIZE,
            all_annotations=all_annotations  # Pass ALL annotations!
        )
        
        all_negative_patches.extend(neg_patches)
        print(f"Extracted {len(neg_patches)} negative patches from {character}")
    
    # Balance the dataset
    num_positive = len(all_positive_patches)
    num_negative = len(all_negative_patches)
    
    print(f"\nTotal positive patches: {num_positive}")
    print(f"Total negative patches: {num_negative}")
    
    # Combine patches
    X = np.array(all_positive_patches + all_negative_patches)
    
    if include_character_labels:
        # For Task 2: Use character-specific labels (0-4) for positive, 5 for negative
        y = np.array(all_positive_labels + [-1] * len(all_negative_patches))
    else:
        # For Task 1: Binary classification (1 for face, 0 for non-face)
        y = np.array([1] * num_positive + [0] * num_negative)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def load_test_images(test_folder: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all test images from a folder.
    
    Args:
        test_folder: Path to test images folder
        
    Returns:
        List of (filename, image) tuples
    """
    images = []
    
    for img_name in sorted(os.listdir(test_folder)):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(test_folder, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                images.append((img_name, image))
    
    return images


if __name__ == "__main__":
    # Test the data loading
    print("Loading training data...")
    X, y = prepare_training_data(include_character_labels=False)
    print(f"Training data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Positive samples: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
