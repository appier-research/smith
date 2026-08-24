"""
### tower_of_hanoi
Generates Tower of Hanoi puzzle tasks

The Tower of Hanoi is a mathematical puzzle consisting of three rods and a number
of disks of different diameters, which can slide onto any rod. The puzzle starts
with the disks stacked on one rod in order of decreasing size. The objective is
to move the entire stack to another rod, following these rules:
1. Only one disk may be moved at a time
2. Each move consists of taking the upper disk from one of the stacks and placing
   it on top of another stack or on an empty rod
3. No disk may be placed on top of a disk that is smaller than it

Default configuration:
```python
min_disks = 3
max_disks = 5
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the Tower of Hanoi puzzle with 3 disks. Move all disks from rod A to rod C using rod B as auxiliary. Provide the sequence of moves in the format: [(from_rod, to_rod), ...]. For example, (A, C) means move the top disk from rod A to rod C.
Answer: [('A', 'C'), ('A', 'B'), ('C', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('A', 'C')]
Metadata: {'source_dataset': 'tower_of_hanoi', 'source_index': 0, 'num_disks': 3, 'source_rod': 'A', 'dest_rod': 'C', 'aux_rod': 'B', 'min_moves': 7, 'difficulty': {'num_disks': (3, 3), 'value': (1, 100)}}

Example 2:
Question: Solve the Tower of Hanoi puzzle with 4 disks. Move all disks from rod A to rod C using rod B as auxiliary. Provide the sequence of moves in the format: [(from_rod, to_rod), ...]. For example, (A, C) means move the top disk from rod A to rod C.
Answer: [('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'B'), ('C', 'A'), ('C', 'B'), ('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'A'), ('C', 'A'), ('B', 'C'), ('A', 'B'), ('A', 'C'), ('B', 'C')]
Metadata: {'source_dataset': 'tower_of_hanoi', 'source_index': 1, 'num_disks': 4, 'source_rod': 'A', 'dest_rod': 'C', 'aux_rod': 'B', 'min_moves': 15, 'difficulty': {'num_disks': (4, 4), 'value': (1, 100)}}

Example 3:
Question: Solve the Tower of Hanoi puzzle with 5 disks. Move all disks from rod A to rod C using rod B as auxiliary. Provide the sequence of moves in the format: [(from_rod, to_rod), ...]. For example, (A, C) means move the top disk from rod A to rod C.
Answer: [('A', 'C'), ('A', 'B'), ('C', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('A', 'C'), ('A', 'B'), ('C', 'B'), ('C', 'A'), ('B', 'A'), ('C', 'B'), ('A', 'C'), ('A', 'B'), ('C', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('A', 'C'), ('B', 'A'), ('C', 'B'), ('C', 'A'), ('B', 'A'), ('B', 'C'), ('A', 'C'), ('A', 'B'), ('C', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('A', 'C')]
Metadata: {'source_dataset': 'tower_of_hanoi', 'source_index': 2, 'num_disks': 5, 'source_rod': 'A', 'dest_rod': 'C', 'aux_rod': 'B', 'min_moves': 31, 'difficulty': {'num_disks': (5, 5), 'value': (1, 100)}}

````
"""
import json
import os
import re
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from processing.gen_description import tool_maker_prompt
from .utils import call_api, eval_dataset, eval_single_instance, dedup_by_question

EVAL_SYSTEM_PROMPT = "You are a helpful assistant."


def solve_tower_of_hanoi(n, source, destination, auxiliary):
    """
    Generate the solution for Tower of Hanoi puzzle.
    Returns a list of moves as tuples (from_rod, to_rod).
    """
    moves = []

    def hanoi(n, source, destination, auxiliary):
        if n == 1:
            moves.append((source, destination))
        else:
            # Move n-1 disks from source to auxiliary using destination
            hanoi(n - 1, source, auxiliary, destination)
            # Move the largest disk from source to destination
            moves.append((source, destination))
            # Move n-1 disks from auxiliary to destination using source
            hanoi(n - 1, auxiliary, destination, source)

    hanoi(n, source, destination, auxiliary)
    return moves


def verify_tower_of_hanoi(moves, num_disks, source='A', dest='C', aux='B'):
    """
    Verify if a sequence of moves correctly solves the Tower of Hanoi puzzle.
    Returns True if valid, False otherwise.
    """
    # Initialize the state: all disks on source rod
    rods = {source: list(range(num_disks, 0, -1)), dest: [], aux: []}

    try:
        for from_rod, to_rod in moves:
            # Check if from_rod has disks
            if not rods[from_rod]:
                return False

            # Get the disk to move
            disk = rods[from_rod].pop()

            # Check if we can place it on to_rod (either empty or on top of larger disk)
            if rods[to_rod] and disk > rods[to_rod][-1]:
                return False

            # Place the disk
            rods[to_rod].append(disk)

        # Check if all disks are now on the destination rod
        return (len(rods[dest]) == num_disks and
                len(rods[source]) == 0 and
                len(rods[aux]) == 0)
    except (KeyError, IndexError):
        return False


def generate_tower_of_hanoi_problem(num_disks, source='A', dest='C', aux='B'):
    """
    Generate a Tower of Hanoi problem with solution.
    """
    question = (
        f"Solve the Tower of Hanoi puzzle with {num_disks} disks. "
        f"Move all disks from rod {source} to rod {dest} using rod {aux} as auxiliary. "
        f"Provide the sequence of moves in the format: [(from_rod, to_rod), ...]. "
        f"For example, ({source}, {dest}) means move the top disk from rod {source} to rod {dest}."
    )

    solution = solve_tower_of_hanoi(num_disks, source, dest, aux)
    answer = str(solution)
    min_moves = 2**num_disks - 1

    metadata = {
        'source_dataset': 'tower_of_hanoi',
        'num_disks': num_disks,
        'source_rod': source,
        'dest_rod': dest,
        'aux_rod': aux,
        'min_moves': min_moves,
        'difficulty': {
            'num_disks': (num_disks, num_disks),
            'value': (1, 100)
        }
    }

    return {
        'question': question,
        'answer': answer,
        'metadata': metadata
    }


def init_data():
    """Initialize and push datasets to HuggingFace Hub"""
    test_data = []

    # Difficulty 2: 3 disks
    random.seed(42)
    for i in range(70):
        problem = generate_tower_of_hanoi_problem(3)
        problem['difficulty'] = 2
        problem['metadata']['source_index'] = i
        test_data.append(problem)

    # Difficulty 3: 4 disks
    for i in range(70):
        problem = generate_tower_of_hanoi_problem(4)
        problem['difficulty'] = 3
        problem['metadata']['source_index'] = i + 70
        test_data.append(problem)

    # Difficulty 4: 5 disks
    for i in range(70):
        problem = generate_tower_of_hanoi_problem(5)
        problem['difficulty'] = 4
        problem['metadata']['source_index'] = i + 140
        test_data.append(problem)

    test_data = dedup_by_question(test_data)
    Dataset.from_list(test_data)  # .push_to_hub("anonymous/tower_of_hanoi", split="test")  # uncomment to upload to HuggingFace Hub

    # Validation data
    val_data = []
    for i in range(120):
        num_disks = random.choice([3, 4])
        problem = generate_tower_of_hanoi_problem(num_disks)
        problem['difficulty'] = 2 if num_disks == 3 else 3
        problem['metadata']['source_index'] = i
        val_data.append(problem)

    val_data = dedup_by_question(val_data)
    Dataset.from_list(val_data)  # .push_to_hub("anonymous/tower_of_hanoi", split="validate")  # uncomment to upload to HuggingFace Hub

    # Training data
    train_data = []
    for i in range(250):
        problem = generate_tower_of_hanoi_problem(3)
        problem['difficulty'] = 2
        problem['metadata']['source_index'] = i
        train_data.append(problem)

    for i in range(250):
        problem = generate_tower_of_hanoi_problem(4)
        problem['difficulty'] = 3
        problem['metadata']['source_index'] = i + 250
        train_data.append(problem)

    train_data = dedup_by_question(train_data)
    Dataset.from_list(train_data)  # .push_to_hub("anonymous/tower_of_hanoi", split="train")  # uncomment to upload to HuggingFace Hub


def load_ds():
    """Load training and test datasets"""
    train = load_dataset("anonymous/tower_of_hanoi", split="validate")
    test = load_dataset("anonymous/tower_of_hanoi", split="test")
    test_a = []
    for t in test:
        t['inputs'] = t['question']
        t['targets'] = t['answer']
        test_a.append(t)

    train_a = []
    for t in train:
        t['inputs'] = t['question']
        t['targets'] = t['answer']
        train_a.append(t)

    return train_a, test_a


def eval_equality(input_text, ground_truth):
    """
    Evaluate if the predicted answer matches the ground truth.
    Handles various formats of move sequences.
    """
    # Extract the answer from the text
    pred_text = input_text.split('FINAL_ANSWER')[-1].replace(':', '').strip()

    # Try to extract list from the prediction
    try:
        # Look for patterns like [('A', 'C'), ('A', 'B'), ...]
        match = re.search(r'\[(.*?)\]', pred_text, re.DOTALL)
        if match:
            pred_text = '[' + match.group(1) + ']'

        # Parse both as Python literals
        import ast
        pred_moves = ast.literal_eval(pred_text)
        ground_moves = ast.literal_eval(ground_truth)

        # Check if they're the same
        if pred_moves == ground_moves:
            return True

        # Also verify if the prediction is a valid solution
        if isinstance(pred_moves, list) and len(pred_moves) > 0:
            # Extract number of disks from ground truth metadata
            num_disks = len(ground_moves) // 2 + 1
            if verify_tower_of_hanoi(pred_moves, num_disks):
                return True
    except (ValueError, SyntaxError, AttributeError):
        pass

    return False


def baseline(eval_ds, client, model_name, system_prompt, experiment_dir=None):
    """Run baseline evaluation without tools"""
    hit, total = 0, 0

    # Setup logging if experiment_dir is provided
    log_file_jsonl = None
    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        log_file_jsonl = os.path.join(experiment_dir, "baseline_evaluations_log.jsonl")
        print(f'Logging results to {log_file_jsonl}')

    # Use ThreadPoolExecutor for parallel evaluation
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks to the thread pool
        future_to_row = {
            executor.submit(eval_single_instance, row, [], client, model_name, None, system_prompt, eval_equality): row
            for row in eval_ds
        }
        # Collect results as they complete with progress bar
        with tqdm(total=len(eval_ds), desc="Evaluating") as pbar:
            for future in as_completed(future_to_row):
                try:
                    correct, tool_used, messages, round_usage, stats = future.result()
                    hit += correct
                    total += 1

                    # Log individual question results if experiment_dir is provided
                    if log_file_jsonl:
                        with open(log_file_jsonl, 'a') as f:
                            # Convert messages to serializable format
                            serializable_messages = []
                            for msg in messages:
                                if hasattr(msg, 'model_dump'):
                                    serializable_messages.append(msg.model_dump())
                                elif hasattr(msg, 'to_dict'):
                                    serializable_messages.append(msg.to_dict())
                                elif isinstance(msg, dict):
                                    serializable_messages.append(msg)
                                else:
                                    serializable_messages.append(str(msg))

                            log_entry = {
                                'stats': stats,
                                'messages': serializable_messages,
                                'round_usage': round_usage
                            }
                            f.write(json.dumps(log_entry) + '\n')

                except Exception as e:
                    print(f"Error evaluating row: {e}")
                    total += 1
                pbar.update(1)

    return hit/total


def tool_making(client, train, experiment_dir, model_name, sample=10, test_size=100, proposal_n=10):
    """Generate and evaluate multiple tool proposals"""
    proposal2results = {}
    for proposal_i in range(proposal_n):
        random.shuffle(train)

        output_dir = os.path.join(experiment_dir, f'attempt_{proposal_i}')
        os.makedirs(output_dir, exist_ok=True)
        full_prompt = tool_maker_prompt
        total_q = sample+test_size
        reference_q = []
        test_q = []
        for idx, row in enumerate(train):
            if len(reference_q) < sample:
                reference_q.append(row)
            elif len(test_q) < test_size:
                test_q.append(row)
            if idx >= total_q:
                break

        questions_text = ""
        for idx, q in enumerate(reference_q):
            questions_text += f"Question {idx+1}: {q['inputs']}\nAnswer: {q['targets']}\n\n"
        current_prompt = full_prompt + questions_text
        res_text, _ = call_api(client, current_prompt, model_name)

        # Extract Python code and JSON from response
        try:
            python_code = re.findall(r'```python\n(.*?)\n```', res_text, re.DOTALL)[0]
            json_code = re.findall(r'```json\n(.*?)\n```', res_text, re.DOTALL)[0]
        except IndexError as e:
            print(e)
            continue

        success = True
        try:
            tool_schema = json.loads(json_code)
        except json.decoder.JSONDecodeError:
            success = False
        if not success:
            continue

        with open(os.path.join(output_dir, f"function.py"), 'w') as f:
            f.write(python_code)

        with open(os.path.join(output_dir, f"function.json"), 'w') as f:
            f.write(json_code)

        if 'functions' in tool_schema:
            tool_schema = tool_schema['functions']
        elif 'tools' in tool_schema:
            tool_schema = tool_schema['tools']
        if isinstance(tool_schema, dict):
            tool_schema = [tool_schema]

        # Transform tool schema to OpenAI format
        openai_tools = []
        for tool in tool_schema:
            openai_tools.append({
                "type": "function",
                "function": tool
            })
        acc, tool_usage_rate = eval_dataset(test_q, client, model_name, openai_tools, os.path.join(output_dir, f"function.py"))
        proposal2results[output_dir] = acc
        print(f"Attempt {proposal_i}: {acc:.3f} accuracy")

    # Sort results and display best
    sorted_results = sorted(proposal2results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== Results Summary ===")
    for path, acc in sorted_results:
        print(f"{path}: {acc:.3f}")

    best_path, best_acc = sorted_results[0]
    print(f"\nBest tool: {best_path} with {best_acc:.3f} accuracy")
    return best_path, best_acc


def run_full_test(test, client, model_name, best_tool_path):
    """Run full evaluation on test set using the best tool"""
    # Load the best tool
    function_py_path = os.path.join(best_tool_path, "function.py")
    function_json_path = os.path.join(best_tool_path, "function.json")

    with open(function_json_path, 'r') as f:
        tool_schema = json.loads(f.read())

    if 'functions' in tool_schema:
        tool_schema = tool_schema['functions']
    elif 'tools' in tool_schema:
        tool_schema = tool_schema['tools']
    if isinstance(tool_schema, dict):
        tool_schema = [tool_schema]

    # Transform to OpenAI format
    openai_tools = []
    for tool in tool_schema:
        openai_tools.append({
            "type": "function",
            "function": tool
        })

    # Run evaluation on full test set
    print(f"\n=== Running full test evaluation ===")
    print(f"Test set size: {len(test)}")
    acc, tool_usage_rate = eval_dataset(test, client, model_name, openai_tools, function_py_path, log_results=True)
    print(f"Final test accuracy: {acc:.3f} ({tool_usage_rate:.3f})")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tower of Hanoi evaluation with tool generation')
    parser.add_argument('--api_key', type=str, default="token-abc121736278", help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default="http://localhost:8087/v1", help='Base URL for API')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Model name to use')
    parser.add_argument('--experiment_dir', type=str, default='experiments/tower_of_hanoi/test', help='Experiment directory')
    parser.add_argument('--baseline', action='store_true', help='Run baseline evaluation without tool generation')
    parser.add_argument('--init', action='store_true', help='Initialize and push datasets to HuggingFace Hub')
    args = parser.parse_args()

    if args.init:
        init_data()
        print("Datasets initialized and pushed to HuggingFace Hub")
    else:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        train, test = load_ds()

        if args.baseline:
            # Run baseline evaluation
            print(f"\nTest set size: {len(test)}")
            test_acc = baseline(test, client, args.model_name, EVAL_SYSTEM_PROMPT, args.experiment_dir)
            print(f"Test accuracy: {test_acc:.3f}")

            print(f"\n=== Final Results ===")
            print(f"Baseline test accuracy: {test_acc:.3f}")
        else:
            # Generate and evaluate multiple tool proposals
            best_tool_path, best_acc = tool_making(client, train, args.experiment_dir, args.model_name)

            # Run full evaluation on test set using best tool
            final_acc = run_full_test(test, client, args.model_name, best_tool_path)

            print(f"\n=== Final Results ===")
            print(f"Best tool validation accuracy: {best_acc:.3f}")
            print(f"Final test accuracy: {final_acc:.3f}")
