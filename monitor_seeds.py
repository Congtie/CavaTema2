"""
Monitor seed sweep testing progress.
"""

import os
from pathlib import Path

SEED_TEST_RESULTS_DIR = Path("seed_test_results")
SEEDS_TO_TEST = [42, 123, 256, 512, 999, 2024]


def monitor_progress():
    """Check progress of seed sweep."""
    print("="*60)
    print("SEED SWEEP PROGRESS")
    print("="*60)

    if not SEED_TEST_RESULTS_DIR.exists():
        print("\nSeed test results directory not created yet.")
        print("Seed sweep is still starting up...")
        return

    print(f"\nChecking results in: {SEED_TEST_RESULTS_DIR.absolute()}")

    completed = 0
    for seed in SEEDS_TO_TEST:
        task1_dir = SEED_TEST_RESULTS_DIR / f"seed_{seed}_task1"
        task2_dir = SEED_TEST_RESULTS_DIR / f"seed_{seed}_task2"

        if task1_dir.exists() and task2_dir.exists():
            print(f"✓ Seed {seed}: COMPLETE")
            completed += 1
        else:
            print(f"◌ Seed {seed}: IN PROGRESS or PENDING")

    print(f"\nProgress: {completed}/{len(SEEDS_TO_TEST)} seeds completed")
    print("\nEstimated time:")
    print(f"  - Each seed: ~1-2 min training + ~3-4 hours inference")
    print(f"  - Total: ~{len(SEEDS_TO_TEST) * 3.5:.1f}-{len(SEEDS_TO_TEST) * 4.5:.1f} hours")

    if completed == len(SEEDS_TO_TEST):
        print("\n✓ All seeds completed! Run evaluate_seeds.py to find the best seed.")


if __name__ == "__main__":
    monitor_progress()
