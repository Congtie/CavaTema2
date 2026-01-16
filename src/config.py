"""
Configuration file for Scooby-Doo Face Detection and Recognition System.
Contains all hyperparameters and paths.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training data paths
TRAIN_PATH = os.path.join(BASE_PATH, "antrenare")
TRAIN_ANNOTATIONS = {
    "fred": os.path.join(TRAIN_PATH, "fred_annotations.txt"),
    "daphne": os.path.join(TRAIN_PATH, "daphne_annotations.txt"),
    "shaggy": os.path.join(TRAIN_PATH, "shaggy_annotations.txt"),
    "velma": os.path.join(TRAIN_PATH, "velma_annotations.txt"),
}
TRAIN_IMAGES = {
    "fred": os.path.join(TRAIN_PATH, "fred"),
    "daphne": os.path.join(TRAIN_PATH, "daphne"),
    "shaggy": os.path.join(TRAIN_PATH, "shaggy"),
    "velma": os.path.join(TRAIN_PATH, "velma"),
}

# Validation data paths
VALIDATION_PATH = os.path.join(BASE_PATH, "validare")
VALIDATION_IMAGES = os.path.join(VALIDATION_PATH, "validare")
VALIDATION_GT_TASK1 = os.path.join(VALIDATION_PATH, "task1_gt_validare.txt")
VALIDATION_GT_TASK2 = {
    "fred": os.path.join(VALIDATION_PATH, "task2_fred_gt_validare.txt"),
    "daphne": os.path.join(VALIDATION_PATH, "task2_daphne_gt_validare.txt"),
    "shaggy": os.path.join(VALIDATION_PATH, "task2_shaggy_gt_validare.txt"),
    "velma": os.path.join(VALIDATION_PATH, "task2_velma_gt_validare.txt"),
}

# Test data paths
TEST_PATH = os.path.join(BASE_PATH, "testare")

# Output paths
OUTPUT_PATH = os.path.join(BASE_PATH, "evaluare", "fisiere_solutie", "solution")
OUTPUT_TASK1 = os.path.join(OUTPUT_PATH, "task1")
OUTPUT_TASK2 = os.path.join(OUTPUT_PATH, "task2")

# Model paths
MODELS_PATH = os.path.join(BASE_PATH, "models")

# =============================================================================
# HOG PARAMETERS
# =============================================================================
# Window size for face patches (must be consistent across training and inference)
WINDOW_SIZE = (64, 64)  # (height, width)

# HOG feature descriptor parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

# =============================================================================
# SLIDING WINDOW PARAMETERS
# =============================================================================
# Minimum window size for sliding window
MIN_WINDOW_SIZE = 32

# Maximum window size for sliding window
MAX_WINDOW_SIZE = 300

# Scale factor for image pyramid
SCALE_FACTOR = 1.2

# Step size for sliding window (in pixels)
STEP_SIZE = 8

# =============================================================================
# SVM PARAMETERS
# =============================================================================
# Regularization parameter for SVM
SVM_C = 1.0

# =============================================================================
# NON-MAXIMUM SUPPRESSION PARAMETERS
# =============================================================================
# IoU threshold for NMS (lower = more aggressive suppression)
NMS_THRESHOLD = 0.2  # Testing more aggressive NMS to reduce false positives

# =============================================================================
# DATA AUGMENTATION PARAMETERS
# =============================================================================
# Number of negative samples per image
NUM_NEGATIVE_SAMPLES_PER_IMAGE = 10

# Minimum overlap for a detection to be considered positive
MIN_POSITIVE_IOU = 0.5

# Maximum overlap for a negative sample (changed from 0.3 to 0.05 to avoid faces in negatives)
MAX_NEGATIVE_IOU = 0.05

# =============================================================================
# DETECTION PARAMETERS
# =============================================================================
# Minimum score threshold for detections (global)
DETECTION_THRESHOLD = 3.0  # Minimum score threshold for face detection

# Per-character thresholds (lower for underrepresented classes)
CHAR_DETECTION_THRESHOLDS = {
    "fred": 3.0,      # Lower - often misses detections
    "daphne": 3.5,    # Standard
    "shaggy": 2.5,    # Lowest - most underrepresented class
    "velma": 3.5,     # Standard - performs well
}

# Characters for Task 2
CHARACTERS = ["fred", "daphne", "shaggy", "velma"]

# Character label mapping
CHAR_TO_LABEL = {
    "fred": 0,
    "daphne": 1,
    "shaggy": 2,
    "velma": 3,
    "unknown": 4,
    "negative": 5,  # Non-face patches
}

LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}
