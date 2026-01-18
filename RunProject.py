import os
import sys

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from run_project import run_project, train_models, load_models
from config import VALIDATION_IMAGES, TEST_PATH, MODELS_PATH

def main(input_folder=None, output_folder=None):
    if input_folder is None:
        input_folder = VALIDATION_IMAGES
    
    run_project(input_folder, output_folder)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args.input, args.output)
