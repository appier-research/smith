"""
Dataset Creation Script
Creates all datasets by importing and calling init_data() from each task module.

Usage:
    python create_all_datasets.py --all
    python create_all_datasets.py --tasks bitwise_arithmetic count_bits
    python create_all_datasets.py --all --force  # Force recreation of existing datasets
"""
import argparse
import sys
from typing import List, Optional
from datasets import load_dataset

# Import all task modules
try:
    from . import ab
    from . import bitwise_arithematic
    from . import count_bits
    from . import polynomial_equations
    from . import polynomial_multiplication
    from . import simple_equations
    from . import simple_integration
    from . import base_conversion
    from . import caesar_cipher
    from . import chain_sum
    from . import gcd
    from . import isomorphic_string
    from . import knights_knaves
    from . import lcm
    from . import prime_factorization
    from . import propositional_logic
    from . import spell_backward
    from . import syllogism
    from . import tower_of_hanoi
    from . import cryptarithm
    from . import calendar_arithmetic
    from . import countdown
except ImportError as e:
    print(f"Error importing module: {e}")
    print("Make sure all required modules are available in the current directory.")
    sys.exit(1)


# Map of task names to their modules
TASK_MODULES = {
    'ab': ab,
    'bitwise_arithmetic': bitwise_arithematic,
    'count_bits': count_bits,
    'polynomial_equations': polynomial_equations,
    'polynomial_multiplication': polynomial_multiplication,
    'simple_equations': simple_equations,
    'simple_integration': simple_integration,
    'base_conversion': base_conversion,
    'caesar_cipher': caesar_cipher,
    'chain_sum': chain_sum,
    'gcd': gcd,
    'isomorphic_string': isomorphic_string,
    'knights_knaves': knights_knaves,
    'lcm': lcm,
    'prime_factorization': prime_factorization,
    'propositional_logic': propositional_logic,
    'spell_backward': spell_backward,
    'syllogism': syllogism,
    'tower_of_hanoi': tower_of_hanoi,
    'cryptarithm': cryptarithm,
    'calendar_arithmetic': calendar_arithmetic,
    'countdown': countdown,
}


def dataset_exists(task_name: str) -> bool:
    """
    Check if dataset already exists on HuggingFace hub.

    Args:
        task_name: Name of the task to check

    Returns:
        True if dataset exists, False otherwise
    """
    hub_name = f"anonymous/{task_name}"
    try:
        # Try to load just the config to check existence
        load_dataset(hub_name, split="train", streaming=True)
        return True
    except (DatasetNotFoundError, Exception):
        return False


def create_dataset(task_name: str, force: bool = False) -> bool:
    """
    Create dataset for a specific task.

    Args:
        task_name: Name of the task to create dataset for
        force: If True, recreate dataset even if it exists

    Returns:
        True if successful, False otherwise, None if skipped
    """
    if task_name not in TASK_MODULES:
        print(f"Error: Unknown task '{task_name}'")
        print(f"Available tasks: {', '.join(TASK_MODULES.keys())}")
        return False

    module = TASK_MODULES[task_name]

    # Check if module has init_data function
    if not hasattr(module, 'init_data'):
        print(f"Warning: Module '{task_name}' does not have init_data() function. Skipping.")
        return False

    # NOTE: push_to_hub is disabled in all task files.
    # The dataset_exists check (HuggingFace lookup) is therefore skipped.
    # Uncomment the block below if you re-enable push_to_hub and want to skip
    # tasks that have already been uploaded:
    # if not force:
    #     if dataset_exists(task_name):
    #         print(f"Dataset '{task_name}' already exists on HuggingFace. Skipping.")
    #         return None

    print(f"\n{'='*60}")
    print(f"Creating dataset for: {task_name}")
    print(f"{'='*60}")

    try:
        module.init_data()
        print(f"✓ Successfully created dataset for {task_name}")
        return True
    except Exception as e:
        print(f"✗ Error creating dataset for {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_all_datasets(tasks: Optional[List[str]] = None, force: bool = False) -> dict:
    """
    Create datasets for all tasks or specified tasks.

    Args:
        tasks: Optional list of task names. If None, creates all datasets.
        force: If True, recreate datasets even if they exist

    Returns:
        Dictionary mapping task names to success status (True/False/None for skipped)
    """
    if tasks is None:
        tasks = list(TASK_MODULES.keys())

    print(f"\n{'='*60}")
    print(f"Starting dataset creation for {len(tasks)} tasks")
    if force:
        print("Force mode: Will recreate existing datasets")
    print(f"{'='*60}\n")

    results = {}
    for task_name in tasks:
        success = create_dataset(task_name, force=force)
        results[task_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [task for task, success in results.items() if success is True]
    failed = [task for task, success in results.items() if success is False]
    skipped = [task for task, success in results.items() if success is None]

    print(f"\nSuccessful: {len(successful)}/{len(tasks)}")
    if successful:
        for task in successful:
            print(f"  ✓ {task}")

    if skipped:
        print(f"\nSkipped (already exists): {len(skipped)}/{len(tasks)}")
        for task in skipped:
            print(f"  ⊘ {task}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(tasks)}")
        for task in failed:
            print(f"  ✗ {task}")

    return results


def list_available_tasks():
    """List all available tasks."""
    print("\nAvailable tasks:")
    for i, task_name in enumerate(sorted(TASK_MODULES.keys()), 1):
        module = TASK_MODULES[task_name]
        has_init = "✓" if hasattr(module, 'init_data') else "✗"
        print(f"  {i:2d}. {task_name:30s} [init_data: {has_init}]")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create datasets for reasoning tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all datasets
  python create_all_datasets.py --all

  # Create specific datasets
  python create_all_datasets.py --tasks bitwise_arithmetic count_bits

  # List available tasks
  python create_all_datasets.py --list
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Create all datasets'
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        help='Specific tasks to create datasets for'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tasks'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of datasets even if they already exist on HuggingFace'
    )

    args = parser.parse_args()

    if args.list:
        list_available_tasks()
        sys.exit(0)

    if args.all:
        results = create_all_datasets(force=args.force)
    elif args.tasks:
        results = create_all_datasets(args.tasks, force=args.force)
    else:
        parser.print_help()
        print("\nPlease specify --all, --tasks, or --list")
        sys.exit(1)

    # Exit with error code if any failed (None = skipped is OK)
    failed_count = sum(1 for success in results.values() if success is False)
    sys.exit(0 if failed_count == 0 else 1)
