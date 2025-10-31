#!/usr/bin/env python3
"""
Continual Learning Testing Script for DuMeNet
Test on all previously learned tasks after each training phase
Usage: python tools/test_continual.py --checkpoint path/to/checkpoint
"""

import os
import sys
import yaml
import argparse
import subprocess
import json
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(current_dir)
sys.path.insert(0, current_dir)
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + os.pathsep + current_dir


def test_single_class(class_name, checkpoint_path, config_path):
    """
    Test on a single class

    Args:
        class_name: Class name to test
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file

    Returns:
        results: Dictionary with test metrics
        success: Test success status
    """
    try:
        print(f"\n{'='*60}")
        print(f"Testing on class: {class_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        # Run testing
        cmd = [
            sys.executable, "-u", "tools/train_val.py",
            "--config", config_path,
            "--class_name", class_name,
            "--checkpoint", checkpoint_path,
            "--test_only",
            "--single_gpu"
        ]

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{class_name}] {output.strip()}")
                output_lines.append(output.strip())

        return_code = process.poll()

        if return_code == 0:
            print(f"✓ Test completed for {class_name}")
            # Parse results from output (you may need to adjust based on your output format)
            results = parse_test_results(output_lines, class_name)
            return results, True
        else:
            print(f"✗ Test failed for {class_name}")
            return None, False

    except Exception as e:
        print(f"Exception during testing {class_name}: {str(e)}")
        return None, False


def parse_test_results(output_lines, class_name):
    """
    Parse test results from output lines
    Extract metrics like AUC, etc.
    """
    results = {
        'class': class_name,
        'pixel_auc': None,
        'image_auc': None,
    }

    # Parse output to extract metrics
    for line in output_lines:
        if 'pixel_auc' in line.lower() or 'pixel auc' in line.lower():
            # Try to extract AUC value
            try:
                # Example: "pixel_auc: 0.95" or "Pixel AUC: 95.0"
                parts = line.split(':')
                if len(parts) >= 2:
                    value = float(parts[-1].strip().replace('%', ''))
                    if value > 1:  # If percentage format
                        value = value / 100.0
                    results['pixel_auc'] = value
            except:
                pass

        if 'image_auc' in line.lower() or 'image auc' in line.lower():
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    value = float(parts[-1].strip().replace('%', ''))
                    if value > 1:
                        value = value / 100.0
                    results['image_auc'] = value
            except:
                pass

    return results


def test_continual_learning(checkpoint_path, learned_tasks, config_path, output_dir):
    """
    Test model on all learned tasks

    Args:
        checkpoint_path: Path to current model checkpoint
        learned_tasks: List of class names that have been learned
        config_path: Path to config file
        output_dir: Directory to save results

    Returns:
        all_results: List of results for each task
    """
    print(f"\n{'='*60}")
    print(f"CONTINUAL LEARNING EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Testing on {len(learned_tasks)} learned tasks")
    print(f"Tasks: {', '.join(learned_tasks)}")
    print(f"{'='*60}")

    all_results = []
    success_count = 0

    for task_idx, class_name in enumerate(learned_tasks):
        print(f"\nTask {task_idx}/{len(learned_tasks)-1}: {class_name}")

        results, success = test_single_class(
            class_name,
            checkpoint_path,
            config_path
        )

        if success and results:
            all_results.append(results)
            success_count += 1
        else:
            all_results.append({
                'class': class_name,
                'pixel_auc': None,
                'image_auc': None,
                'error': 'Test failed'
            })

    # Compute average metrics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    valid_pixel_aucs = [r['pixel_auc'] for r in all_results if r['pixel_auc'] is not None]
    valid_image_aucs = [r['image_auc'] for r in all_results if r['image_auc'] is not None]

    if valid_pixel_aucs:
        avg_pixel_auc = sum(valid_pixel_aucs) / len(valid_pixel_aucs)
        print(f"Average Pixel AUC: {avg_pixel_auc:.4f}")

    if valid_image_aucs:
        avg_image_auc = sum(valid_image_aucs) / len(valid_image_aucs)
        print(f"Average Image AUC: {avg_image_auc:.4f}")

    print(f"\nPer-class results:")
    for result in all_results:
        pixel_auc_str = f"{result['pixel_auc']:.4f}" if result['pixel_auc'] else "N/A"
        image_auc_str = f"{result['image_auc']:.4f}" if result['image_auc'] else "N/A"
        print(f"  {result['class']:15s}: Pixel={pixel_auc_str}, Image={image_auc_str}")

    print(f"\nSuccess rate: {success_count}/{len(learned_tasks)}")

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'continual_test_results.json')

    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': checkpoint_path,
            'learned_tasks': learned_tasks,
            'results': all_results,
            'avg_pixel_auc': avg_pixel_auc if valid_pixel_aucs else None,
            'avg_image_auc': avg_image_auc if valid_image_aucs else None,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Testing for DuMeNet')
    parser.add_argument('--config', type=str, default='tools/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut'],
                        help='List of tasks/classes to test (in learning order)')
    parser.add_argument('--output_dir', type=str, default='results_continual',
                        help='Directory to save test results')
    args = parser.parse_args()

    print("DuMeNet Continual Learning Testing")
    print("="*60)

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        sys.exit(1)

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)

    # Run continual learning evaluation
    test_continual_learning(
        checkpoint_path=args.checkpoint,
        learned_tasks=args.tasks,
        config_path=args.config,
        output_dir=args.output_dir
    )

    print("\nContinual learning evaluation completed!")


if __name__ == "__main__":
    main()