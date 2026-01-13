"""
Scooby-Doo Face Detection and Recognition System

This package implements a classical computer vision pipeline for:
- Task 1: Face detection using HOG + SVM + Sliding Window
- Task 2: Character recognition (Fred, Daphne, Shaggy, Velma)
"""

from .config import *
from .data_loader import prepare_training_data, load_test_images
from .feature_extraction import extract_hog_features, extract_combined_features
from .classifier import FaceDetector, CharacterRecognizer
from .sliding_window import detect_faces_multiscale, detect_and_classify
from .nms import non_maximum_suppression_fast, apply_nms_to_results
from .run_project import run_project, train_models, load_models

__version__ = "1.0.0"
__author__ = "Student"
