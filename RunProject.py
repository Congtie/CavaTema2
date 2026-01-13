"""
Simple wrapper script to run the Scooby-Doo Face Detection and Recognition System.
This is the main entry point for the project.
"""

import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from run_project import run_project, train_models, load_models
from config import VALIDATION_IMAGES, TEST_PATH, MODELS_PATH


def main(input_folder=None, output_folder=None):
    """
    Run the complete face detection and recognition pipeline.
    
    Args:
        input_folder: Path to folder with test images (default: validare/validare)
        output_folder: Path for output files (default: evaluare/fisiere_solutie/solution)
    """
    if input_folder is None:
        # Default to validation images
        input_folder = VALIDATION_IMAGES
    
    run_project(input_folder, output_folder)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Scooby-Doo Face Detection")
    parser.add_argument("--input", type=str, default=None, 
                        help="Input folder with test images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder for results")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
