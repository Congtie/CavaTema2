import os
import sys
import json
import numpy as np
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_PATH, VALIDATION_IMAGES, RANDOM_SEED
from run_project import train_models, load_models, run_inference, save_results

SEEDS_TO_TEST = [42, 123, 256, 512, 999, 2024]

TEMP_MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_models")


def modify_config_seed(seed_value):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    import re
    new_content = re.sub(
        r'RANDOM_SEED = None',
        f'RANDOM_SEED = {seed_value}',
        content
    )

    with open(config_path, 'w') as f:
        f.write(new_content)


def restore_config_seed():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    import re
    new_content = re.sub(
        r'RANDOM_SEED = \d+',
        'RANDOM_SEED = None',
        content
    )

    with open(config_path, 'w') as f:
        f.write(new_content)


def test_seed(seed_value, output_dir):
    print(f"\n{'='*60}")
    print(f"Testing RANDOM_SEED = {seed_value}")
    print(f"{'='*60}")

    modify_config_seed(seed_value)

    import importlib
    import config
    importlib.reload(config)

    if os.path.exists(TEMP_MODELS_PATH):
        shutil.rmtree(TEMP_MODELS_PATH)
    os.makedirs(TEMP_MODELS_PATH, exist_ok=True)

    try:
        print(f"\n[1/3] Training models with seed {seed_value}...")
        face_detector, char_recognizer = train_models(TEMP_MODELS_PATH)

        print(f"\n[2/3] Running inference with seed {seed_value}...")
        task1_results, task2_results = run_inference(
            VALIDATION_IMAGES,
            face_detector,
            char_recognizer
        )

        print(f"\n[3/3] Saving results for seed {seed_value}...")
        seed_output_task1 = os.path.join(output_dir, f"seed_{seed_value}_task1")
        seed_output_task2 = os.path.join(output_dir, f"seed_{seed_value}_task2")
        save_results(task1_results, task2_results, seed_output_task1, seed_output_task2)

        print(f"\nResults saved for seed {seed_value}")

        metadata = {
            "seed": seed_value,
            "task1_detections": len(task1_results['detections']),
            "task1_output": seed_output_task1,
            "task2_output": seed_output_task2
        }

        return metadata

    except Exception as e:
        print(f"\nError testing seed {seed_value}: {str(e)}")
        return None


def run_seed_sweep():
    print("="*60)
    print("RANDOM SEED SWEEP TESTING")
    print("="*60)
    print(f"\nTesting seeds: {SEEDS_TO_TEST}")
    print(f"Validation images: {VALIDATION_IMAGES}")

    output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seed_test_results")
    os.makedirs(output_base, exist_ok=True)

    results = []

    for i, seed in enumerate(SEEDS_TO_TEST, 1):
        print(f"\n[{i}/{len(SEEDS_TO_TEST)}] Testing seed {seed}...")
        metadata = test_seed(seed, output_base)
        if metadata:
            results.append(metadata)

    restore_config_seed()

    summary_path = os.path.join(output_base, "seeds_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SEED SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults summary saved to: {summary_path}")
    print(f"Detailed results in: {output_base}")
    print("\nNext step: Use evalueaza_solutie.py to evaluate each seed's results")
    print("Then select the seed with the best AP/mAP scores")


if __name__ == "__main__":
    run_seed_sweep()
