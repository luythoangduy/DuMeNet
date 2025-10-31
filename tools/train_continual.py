#!/usr/bin/env python3
"""
Continual Learning Training Script for DuMeNet
Sequential training on multiple classes/tasks
Usage: python tools/train_continual.py --config tools/config.yaml
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
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
            print(f"Task {task_id} ({class_name}): Training completed")
            return checkpoint_path, True
        else:
            print(f"Task {task_id} ({class_name}): Training failed")
            return None, False

    except Exception as e:
        print(f"Task {task_id} ({class_name}): Exception - {str(e)}")
        return None, False


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Training for DuMeNet')
    parser.add_argument('--config', type=str, default='tools/config.yaml',
                        help='Path to config file')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['hazelnut', 'bottle', 'cable','capsule', 'metal_nut', 'pill', 'toothbrush', 'transistor', 'zipper', 'screw'],
                        help='List of tasks/classes to train sequentially')
    parser.add_argument('--resume_from_task', type=int, default=0,
                        help='Resume from specific task (0-indexed)')
    args = parser.parse_args()

    print("DuMeNet Continual Learning Training")
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

    # Start continual learning
    start_time = time.time()
    success_count = 0
    prev_checkpoint = None

    for task_id in range(args.resume_from_task, len(task_sequence)):
        class_name = task_sequence[task_id]

        # Train on this task
        checkpoint_path, success = train_task(
            task_id,
            class_name,
            base_config,
            prev_checkpoint
        )

        if success:
            success_count += 1
            prev_checkpoint = checkpoint_path
        else:
            print(f"\nWarning: Task {task_id} failed, continuing without checkpoint")

        # Short rest between tasks
        if task_id < len(task_sequence) - 1:
            print(f"\nResting 10 seconds before next task...")
            time.sleep(10)

    # Summary
    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print("CONTINUAL LEARNING SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks completed: {success_count}/{len(task_sequence)}")
    print(f"Total time: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"Average per task: {duration/len(task_sequence)/60:.1f} minutes")

    if success_count == len(task_sequence):
        print("✓ All tasks completed successfully!")
    else:
        failed = len(task_sequence) - success_count
        print(f"✗ {failed} tasks failed")

    print("\nContinual learning completed!")


if __name__ == "__main__":
    main()