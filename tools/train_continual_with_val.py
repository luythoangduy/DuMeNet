#!/usr/bin/env python3
"""
Continual Learning Training Script with Validation for DuMeNet
Sequential training on multiple classes/tasks with validation after each task
Usage: python tools/train_continual_with_val.py --config tools/config.yaml
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
import json
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(current_dir)
sys.path.insert(0, current_dir)
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + os.pathsep + current_dir


def train_task(task_id, class_name, base_config, prev_checkpoint=None):
    """
    Train a single task in continual learning sequence

    Args:
        task_id: Task index (0, 1, 2, ...)
        class_name: Class name to train on
        base_config: Base configuration
        prev_checkpoint: Path to checkpoint from previous task (for continual learning)

    Returns:
        checkpoint_path: Path to saved checkpoint
        success: Training success status
    """
    try:
        import copy

        # Create config for this task
        task_config = copy.deepcopy(base_config)
        task_config['wandb']['name'] = f"continual_task{task_id}_{class_name}"
        task_config['wandb']['project'] = "UniAD-Continual"

        # Load from previous checkpoint if available
        if prev_checkpoint is not None and os.path.exists(prev_checkpoint):
            task_config['saver']['load_path'] = prev_checkpoint
            print(f"Loading from previous checkpoint: {prev_checkpoint}")

        # Save task config
        config_path = f"config_task{task_id}_{class_name}_temp.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False)

        print(f"\n{'='*60}")
        print(f"Task {task_id}: Training on {class_name}")
        print(f"{'='*60}")

        # Run training
        cmd = [
            sys.executable, "-u", "tools/train_val.py",
            "--config", config_path,
            "--class_name", class_name,
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

        # Print real-time output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[Task{task_id}-{class_name}] {output.strip()}")

        return_code = process.poll()

        # Find checkpoint path
        checkpoint_dir = os.path.join(
            task_config['saver']['save_dir'],
            class_name
        )
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt_best.pth.tar")

        # Clean up temp config
        try:
            os.remove(config_path)
        except:
            pass

        if return_code == 0:
            print(f"✓ Task {task_id} ({class_name}): Training completed")
            return checkpoint_path, True
        else:
            print(f"✗ Task {task_id} ({class_name}): Training failed")
            return None, False

    except Exception as e:
        print(f"✗ Task {task_id} ({class_name}): Exception - {str(e)}")
        return None, False


def validate_all_tasks(checkpoint_path, learned_tasks, base_config, task_id):
    """
    Validate on all learned tasks so far

    Args:
        checkpoint_path: Path to current model checkpoint
        learned_tasks: List of class names that have been learned so far
        base_config: Base configuration
        task_id: Current task ID

    Returns:
        results: Dictionary with validation results for all tasks
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION AFTER TASK {task_id}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validating on {len(learned_tasks)} learned tasks: {', '.join(learned_tasks)}")
    print(f"{'='*60}")

    results = {
        'task_id': task_id,
        'checkpoint': checkpoint_path,
        'learned_tasks': learned_tasks.copy(),
        'per_class_results': [],
        'avg_pixel_auc': None,
        'avg_image_auc': None,
    }

    all_pixel_aucs = []
    all_image_aucs = []

    for class_idx, class_name in enumerate(learned_tasks):
        print(f"\n[{class_idx+1}/{len(learned_tasks)}] Validating on: {class_name}")

        # Test on this class
        class_result = validate_single_class(
            class_name,
            checkpoint_path,
            base_config
        )

        results['per_class_results'].append(class_result)

        if class_result.get('pixel_auc') is not None:
            all_pixel_aucs.append(class_result['pixel_auc'])
        if class_result.get('image_auc') is not None:
            all_image_aucs.append(class_result['image_auc'])

    # Compute averages
    if all_pixel_aucs:
        results['avg_pixel_auc'] = sum(all_pixel_aucs) / len(all_pixel_aucs)
    if all_image_aucs:
        results['avg_image_auc'] = sum(all_image_aucs) / len(all_image_aucs)

    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY - After Task {task_id}")
    print(f"{'='*60}")

    for class_result in results['per_class_results']:
        pixel_str = f"{class_result.get('pixel_auc', 0):.4f}" if class_result.get('pixel_auc') else "N/A"
        image_str = f"{class_result.get('image_auc', 0):.4f}" if class_result.get('image_auc') else "N/A"
        print(f"  {class_result['class']:15s}: Pixel={pixel_str}, Image={image_str}")

    if results['avg_pixel_auc']:
        print(f"\nAverage Pixel AUC: {results['avg_pixel_auc']:.4f}")
    if results['avg_image_auc']:
        print(f"Average Image AUC: {results['avg_image_auc']:.4f}")

    print(f"{'='*60}")

    return results


def validate_single_class(class_name, checkpoint_path, base_config):
    """
    Validate on a single class

    Args:
        class_name: Class name to validate
        checkpoint_path: Path to model checkpoint
        base_config: Base configuration

    Returns:
        result: Dictionary with validation metrics
    """
    try:
        import copy

        # Create temp config
        temp_config = copy.deepcopy(base_config)
        temp_config['saver']['load_path'] = checkpoint_path

        config_path = f"config_val_{class_name}_temp.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(temp_config, f, default_flow_style=False)

        # Run validation (test mode)
        cmd = [
            sys.executable, "-u", "tools/train_val.py",
            "--config", config_path,
            "--class_name", class_name,
            "--test_only",
            "--single_gpu"
        ]

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
                output_lines.append(output.strip())
                # Don't print every line to reduce clutter

        return_code = process.poll()

        # Clean up
        try:
            os.remove(config_path)
        except:
            pass

        # Parse results
        result = {
            'class': class_name,
            'pixel_auc': None,
            'image_auc': None,
            'success': return_code == 0
        }

        # Extract metrics from output
        for line in output_lines:
            line_lower = line.lower()
            if 'pixel' in line_lower and 'auc' in line_lower:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value = float(parts[-1].strip().replace('%', '').replace(',', ''))
                        if value > 1:
                            value = value / 100.0
                        result['pixel_auc'] = value
                except:
                    pass

            if 'image' in line_lower and 'auc' in line_lower:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value = float(parts[-1].strip().replace('%', '').replace(',', ''))
                        if value > 1:
                            value = value / 100.0
                        result['image_auc'] = value
                except:
                    pass

        if result['success']:
            print(f"  ✓ {class_name}: Pixel={result.get('pixel_auc', 'N/A')}, Image={result.get('image_auc', 'N/A')}")
        else:
            print(f"  ✗ {class_name}: Validation failed")

        return result

    except Exception as e:
        print(f"  ✗ {class_name}: Exception - {str(e)}")
        return {
            'class': class_name,
            'pixel_auc': None,
            'image_auc': None,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Training with Validation for DuMeNet')
    parser.add_argument('--config', type=str, default='tools/config.yaml',
                        help='Path to config file')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut'],
                        help='List of tasks/classes to train sequentially')
    parser.add_argument('--resume_from_task', type=int, default=0,
                        help='Resume from specific task (0-indexed)')
    parser.add_argument('--output_dir', type=str, default='results_continual',
                        help='Directory to save validation results')
    args = parser.parse_args()

    print("DuMeNet Continual Learning Training with Validation")
    print("="*60)

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)

    # Load base config
    try:
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
        print(f"Loaded config: {args.config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Task sequence
    task_sequence = args.tasks
    print(f"\nTask sequence ({len(task_sequence)} tasks):")
    for i, task in enumerate(task_sequence):
        print(f"  Task {i}: {task}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start continual learning
    start_time = time.time()
    success_count = 0
    prev_checkpoint = None
    all_validation_results = []

    for task_id in range(args.resume_from_task, len(task_sequence)):
        class_name = task_sequence[task_id]

        # Train on this task
        print(f"\n{'#'*60}")
        print(f"# TASK {task_id}: {class_name}")
        print(f"{'#'*60}")

        checkpoint_path, success = train_task(
            task_id,
            class_name,
            base_config,
            prev_checkpoint
        )

        if success and checkpoint_path and os.path.exists(checkpoint_path):
            success_count += 1
            prev_checkpoint = checkpoint_path

            # Validate on all learned tasks so far
            learned_tasks_so_far = task_sequence[:task_id+1]
            validation_results = validate_all_tasks(
                checkpoint_path,
                learned_tasks_so_far,
                base_config,
                task_id
            )

            all_validation_results.append(validation_results)

            # Save intermediate results
            results_file = os.path.join(args.output_dir, f'validation_after_task{task_id}.json')
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            print(f"\n✓ Validation results saved to: {results_file}")

        else:
            print(f"\n✗ Task {task_id} failed, stopping continual learning")
            break

        # Short rest between tasks
        if task_id < len(task_sequence) - 1:
            print(f"\nResting 10 seconds before next task...")
            time.sleep(10)

    # Final summary
    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print("CONTINUAL LEARNING FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks completed: {success_count}/{len(task_sequence)}")
    print(f"Total time: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"Average per task: {duration/success_count/60:.1f} minutes" if success_count > 0 else "N/A")

    # Save all results
    final_results_file = os.path.join(args.output_dir, 'continual_learning_all_results.json')
    with open(final_results_file, 'w') as f:
        json.dump({
            'task_sequence': task_sequence,
            'completed_tasks': success_count,
            'total_time_seconds': duration,
            'validation_results': all_validation_results
        }, f, indent=2)

    print(f"\n✓ All results saved to: {final_results_file}")

    # Print forgetting analysis if we have multiple tasks
    if len(all_validation_results) > 1:
        print(f"\n{'='*60}")
        print("FORGETTING ANALYSIS")
        print(f"{'='*60}")

        # For each task, track performance over time
        for task_idx in range(len(task_sequence[:success_count])):
            task_name = task_sequence[task_idx]
            performances = []

            for val_result in all_validation_results[task_idx:]:
                for class_result in val_result['per_class_results']:
                    if class_result['class'] == task_name:
                        if class_result.get('pixel_auc') is not None:
                            performances.append(class_result['pixel_auc'])
                        break

            if len(performances) > 1:
                best_perf = max(performances)
                final_perf = performances[-1]
                forgetting = best_perf - final_perf
                print(f"  {task_name:15s}: Best={best_perf:.4f}, Final={final_perf:.4f}, Forgetting={forgetting:.4f}")

    if success_count == len(task_sequence):
        print(f"\n{'='*60}")
        print("✓ All tasks completed successfully!")
        print(f"{'='*60}")
    else:
        failed = len(task_sequence) - success_count
        print(f"\n✗ {failed} tasks failed or not completed")

    print("\nContinual learning with validation completed!")


if __name__ == "__main__":
    main()