"""
Main pipeline script for Scooby-Doo Face Detection and Recognition.
Run this script to train models, run inference, and generate output files.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_PATH, VALIDATION_IMAGES, TEST_PATH,
    OUTPUT_PATH, OUTPUT_TASK1, OUTPUT_TASK2,
    MODELS_PATH, CHARACTERS,
    MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, SCALE_FACTOR, STEP_SIZE,
    DETECTION_THRESHOLD, NMS_THRESHOLD, WINDOW_SIZE
)
from data_loader import prepare_training_data, load_test_images
from classifier import FaceDetector, CharacterRecognizer
from sliding_window import detect_faces_multiscale, detect_and_classify
from nms import non_maximum_suppression_fast, apply_nms_to_results


def train_models(save_path: str = MODELS_PATH):
    """
    Train both face detector and character recognizer.
    
    Args:
        save_path: Path to save trained models
    """
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    # Prepare training data for Task 1 (binary face/non-face)
    print("\n[1/4] Preparing training data for face detection...")
    X_task1, y_task1 = prepare_training_data(include_character_labels=False)
    print(f"Task 1 data: {X_task1.shape[0]} samples")
    print(f"  - Positive (faces): {np.sum(y_task1 == 1)}")
    print(f"  - Negative (non-faces): {np.sum(y_task1 == 0)}")
    
    # Prepare training data for Task 2 (character recognition)
    print("\n[2/4] Preparing training data for character recognition...")
    X_task2, y_task2 = prepare_training_data(include_character_labels=True)
    print(f"Task 2 data: {X_task2.shape[0]} samples")
    
    # Train face detector
    print("\n[3/4] Training Face Detector (Task 1)...")
    face_detector = FaceDetector(use_color=False)
    face_detector.train(X_task1, y_task1, validate=True)
    
    # Train character recognizer
    print("\n[4/4] Training Character Recognizer (Task 2)...")
    char_recognizer = CharacterRecognizer(use_color=True)
    char_recognizer.train(X_task2, y_task2, validate=True)
    
    # Save models
    os.makedirs(save_path, exist_ok=True)
    face_detector.save(os.path.join(save_path, "face_detector.pkl"))
    char_recognizer.save(os.path.join(save_path, "character_recognizer.pkl"))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return face_detector, char_recognizer


def load_models(model_path: str = MODELS_PATH) -> Tuple[FaceDetector, CharacterRecognizer]:
    """
    Load trained models.
    
    Args:
        model_path: Path to saved models
        
    Returns:
        (face_detector, character_recognizer) tuple
    """
    face_detector = FaceDetector()
    face_detector.load(os.path.join(model_path, "face_detector.pkl"))
    
    char_recognizer = CharacterRecognizer()
    char_recognizer.load(os.path.join(model_path, "character_recognizer.pkl"))
    
    return face_detector, char_recognizer


def run_inference(
    image_folder: str,
    face_detector: FaceDetector,
    char_recognizer: CharacterRecognizer,
    detection_threshold: float = DETECTION_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD
) -> Tuple[Dict, Dict]:
    """
    Run inference on all images in a folder.
    
    Args:
        image_folder: Path to folder containing test images
        face_detector: Trained face detector
        char_recognizer: Trained character recognizer
        detection_threshold: Score threshold for detections
        nms_threshold: IoU threshold for NMS
        
    Returns:
        (task1_results, task2_results) dictionaries
    """
    print("=" * 60)
    print("INFERENCE PHASE")
    print("=" * 60)
    
    # Load test images
    print(f"\nLoading images from: {image_folder}")
    images = load_test_images(image_folder)
    print(f"Found {len(images)} images")
    
    # Initialize results
    task1_results = {
        'detections': [],
        'scores': [],
        'file_names': []
    }
    
    task2_results = {char: {
        'detections': [],
        'scores': [],
        'file_names': []
    } for char in CHARACTERS}
    
    # Process each image
    for img_name, image in tqdm(images, desc="Processing images"):
        # Detect and classify
        results = detect_and_classify(
            image, face_detector, char_recognizer,
            min_window=MIN_WINDOW_SIZE,
            max_window=MAX_WINDOW_SIZE,
            scale_factor=SCALE_FACTOR,
            step_size=STEP_SIZE,
            threshold=detection_threshold,
            show_progress=False
        )
        
        # Apply NMS
        results = apply_nms_to_results(results, nms_threshold)
        
        # Collect Task 1 results (all faces)
        for xmin, ymin, xmax, ymax, score in results['all_faces']:
            task1_results['detections'].append([xmin, ymin, xmax, ymax])
            task1_results['scores'].append(score)
            task1_results['file_names'].append(img_name)
        
        # Collect Task 2 results (per character)
        for char_name in CHARACTERS:
            for xmin, ymin, xmax, ymax, score in results['characters'][char_name]:
                task2_results[char_name]['detections'].append([xmin, ymin, xmax, ymax])
                task2_results[char_name]['scores'].append(score)
                task2_results[char_name]['file_names'].append(img_name)
    
    print(f"\nTask 1: {len(task1_results['detections'])} total face detections")
    for char_name in CHARACTERS:
        print(f"Task 2 - {char_name}: {len(task2_results[char_name]['detections'])} detections")
    
    return task1_results, task2_results


def save_results(
    task1_results: Dict,
    task2_results: Dict,
    output_path_task1: str = OUTPUT_TASK1,
    output_path_task2: str = OUTPUT_TASK2
):
    """
    Save results to .npy files for evaluation.
    
    Args:
        task1_results: Results dictionary for Task 1
        task2_results: Results dictionary for Task 2
        output_path_task1: Output folder for Task 1
        output_path_task2: Output folder for Task 2
    """
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(output_path_task1, exist_ok=True)
    os.makedirs(output_path_task2, exist_ok=True)
    
    # Save Task 1 results
    print(f"\nSaving Task 1 results to: {output_path_task1}")
    np.save(
        os.path.join(output_path_task1, "detections_all_faces.npy"),
        np.array(task1_results['detections'])
    )
    np.save(
        os.path.join(output_path_task1, "scores_all_faces.npy"),
        np.array(task1_results['scores'])
    )
    np.save(
        os.path.join(output_path_task1, "file_names_all_faces.npy"),
        np.array(task1_results['file_names'])
    )
    
    # Save Task 2 results
    print(f"Saving Task 2 results to: {output_path_task2}")
    for char_name in CHARACTERS:
        np.save(
            os.path.join(output_path_task2, f"detections_{char_name}.npy"),
            np.array(task2_results[char_name]['detections'])
        )
        np.save(
            os.path.join(output_path_task2, f"scores_{char_name}.npy"),
            np.array(task2_results[char_name]['scores'])
        )
        np.save(
            os.path.join(output_path_task2, f"file_names_{char_name}.npy"),
            np.array(task2_results[char_name]['file_names'])
        )
    
    print("\nResults saved successfully!")


def visualize_detections(
    image_path: str,
    face_detector: FaceDetector,
    char_recognizer: CharacterRecognizer,
    output_path: str = None
):
    """
    Visualize detections on a single image.
    
    Args:
        image_path: Path to image
        face_detector: Trained face detector
        char_recognizer: Trained character recognizer
        output_path: Optional path to save visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Run detection and classification
    results = detect_and_classify(
        image, face_detector, char_recognizer,
        min_window=MIN_WINDOW_SIZE,
        max_window=MAX_WINDOW_SIZE,
        scale_factor=SCALE_FACTOR,
        step_size=STEP_SIZE,
        threshold=DETECTION_THRESHOLD,
        show_progress=True
    )
    
    # Apply NMS
    results = apply_nms_to_results(results, NMS_THRESHOLD)
    
    # Colors for each character
    colors = {
        'fred': (255, 165, 0),    # Orange
        'daphne': (128, 0, 128),  # Purple
        'shaggy': (0, 128, 0),    # Green
        'velma': (0, 0, 255),     # Red
    }
    
    # Draw detections
    vis_image = image.copy()
    
    for char_name in CHARACTERS:
        color = colors[char_name]
        for xmin, ymin, xmax, ymax, score in results['characters'][char_name]:
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{char_name}: {score:.2f}"
            cv2.putText(vis_image, label, (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.imshow("Detections", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_project(input_folder_name: str, output_folder: str = None):
    """
    Main function to run the complete pipeline.
    
    Args:
        input_folder_name: Path to folder containing test images
        output_folder: Optional custom output folder
    """
    # Set output paths
    if output_folder:
        output_task1 = os.path.join(output_folder, "task1")
        output_task2 = os.path.join(output_folder, "task2")
    else:
        output_task1 = OUTPUT_TASK1
        output_task2 = OUTPUT_TASK2
    
    # Check if models exist
    face_detector_path = os.path.join(MODELS_PATH, "face_detector.pkl")
    char_recognizer_path = os.path.join(MODELS_PATH, "character_recognizer.pkl")
    
    if os.path.exists(face_detector_path) and os.path.exists(char_recognizer_path):
        print("Loading pre-trained models...")
        face_detector, char_recognizer = load_models(MODELS_PATH)
    else:
        print("No pre-trained models found. Training new models...")
        face_detector, char_recognizer = train_models(MODELS_PATH)
    
    # Run inference
    task1_results, task2_results = run_inference(
        input_folder_name,
        face_detector,
        char_recognizer,
        detection_threshold=DETECTION_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )
    
    # Save results
    save_results(task1_results, task2_results, output_task1, output_task2)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to:")
    print(f"  Task 1: {output_task1}")
    print(f"  Task 2: {output_task2}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Scooby-Doo Face Detection and Recognition System"
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test', 'validate', 'full'],
        default='full',
        help='Mode: train (only train), test (only inference), validate (run on validation set), full (train + inference)'
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='Input folder containing test images'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output folder for results'
    )
    parser.add_argument(
        '--visualize', type=str, default=None,
        help='Path to image for visualization'
    )
    
    args = parser.parse_args()
    
    if args.visualize:
        # Visualization mode
        face_detector, char_recognizer = load_models(MODELS_PATH)
        visualize_detections(args.visualize, face_detector, char_recognizer)
        return
    
    if args.mode == 'train':
        # Training only
        train_models(MODELS_PATH)
        
    elif args.mode == 'validate':
        # Run on validation set
        run_project(VALIDATION_IMAGES, args.output)
        
    elif args.mode == 'test':
        # Run on test set
        input_folder = args.input or TEST_PATH
        run_project(input_folder, args.output)
        
    else:  # full
        # Train and run inference
        input_folder = args.input or VALIDATION_IMAGES
        run_project(input_folder, args.output)


if __name__ == "__main__":
    main()
