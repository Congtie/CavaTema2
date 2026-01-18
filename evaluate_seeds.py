"""
Evaluate seed sweep results.
Runs evalueaza_solutie.py on each seed's results and compares scores.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

SEED_TEST_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "seed_test_results"
)

EVALUARE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluare",
    "cod_evaluare",
    "evalueaza_solutie.py"
)


def run_evaluation_for_seed(seed_value):
    """Run evaluation script for a specific seed."""
    print(f"\n{'='*60}")
    print(f"Evaluating Seed {seed_value}")
    print(f"{'='*60}")

    task1_dir = os.path.join(SEED_TEST_RESULTS_DIR, f"seed_{seed_value}_task1")
    task2_dir = os.path.join(SEED_TEST_RESULTS_DIR, f"seed_{seed_value}_task2")

    if not os.path.exists(task1_dir) or not os.path.exists(task2_dir):
        print(f"Results directories not found for seed {seed_value}")
        print(f"  Task 1: {task1_dir}")
        print(f"  Task 2: {task2_dir}")
        return None

    try:
        # Run evaluation
        result = subprocess.run(
            [sys.executable, EVALUARE_SCRIPT, task1_dir, task2_dir],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(EVALUARE_SCRIPT)
        )

        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)

        # Try to extract scores from output
        output = result.stdout
        scores = {}

        if "AP (average precision)" in output:
            # Extract Task 1 AP
            for line in output.split('\n'):
                if "AP (average precision)" in line and "Task 1" in output:
                    try:
                        ap_value = float(line.split()[-1].rstrip('%'))
                        scores['task1_ap'] = ap_value
                    except:
                        pass

        if "TOTAL SCORE" in output:
            try:
                for line in output.split('\n'):
                    if "TOTAL SCORE" in line:
                        score_value = float(line.split()[-1])
                        scores['total_score'] = score_value
                        break
            except:
                pass

        return {
            'seed': seed_value,
            'scores': scores,
            'full_output': result.stdout
        }

    except Exception as e:
        print(f"Error evaluating seed {seed_value}: {str(e)}")
        return None


def main():
    """Main evaluation function."""
    print("="*60)
    print("SEED SWEEP EVALUATION")
    print("="*60)

    if not os.path.exists(SEED_TEST_RESULTS_DIR):
        print(f"\nResults directory not found: {SEED_TEST_RESULTS_DIR}")
        print("Waiting for seed sweep to complete...")
        return

    print(f"\nResults directory: {SEED_TEST_RESULTS_DIR}")
    print(f"Evaluation script: {EVALUARE_SCRIPT}")

    # Find all seed results
    seed_results = {}

    for item in sorted(os.listdir(SEED_TEST_RESULTS_DIR)):
        if item.startswith("seed_") and item.endswith("_task1"):
            try:
                seed_value = int(item.replace("seed_", "").replace("_task1", ""))
                result = run_evaluation_for_seed(seed_value)
                if result:
                    seed_results[seed_value] = result
            except:
                pass

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    if not seed_results:
        print("\nNo seed results found to evaluate yet.")
        print("Wait for seed sweep to complete.")
        return

    # Save summary
    summary_file = os.path.join(SEED_TEST_RESULTS_DIR, "evaluation_results.json")
    with open(summary_file, 'w') as f:
        json.dump(seed_results, f, indent=2)

    print(f"\nEvaluation results saved to: {summary_file}")
    print("\nEvaluated seeds:")
    for seed in sorted(seed_results.keys()):
        print(f"  Seed {seed}: {seed_results[seed]['scores']}")

    # Find best seed
    best_seed = None
    best_score = 0

    for seed, result in seed_results.items():
        if 'total_score' in result['scores']:
            score = result['scores']['total_score']
            if score > best_score:
                best_score = score
                best_seed = seed

    if best_seed:
        print(f"\nBest seed: {best_seed} with score {best_score}/10")


if __name__ == "__main__":
    main()
