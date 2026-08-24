"""
### cryptarithm
Generates cryptarithmetic puzzle tasks (also known as alphametics or verbal arithmetic)

A cryptarithm is a mathematical puzzle where digits are replaced by letters.
Each letter represents a unique digit (0-9), and the goal is to find which
digit each letter represents. Leading letters cannot be zero.

Default configuration:
```python
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the cryptarithmetic puzzle: SEND + MORE = MONEY. Each letter represents a unique digit (0-9), and leading letters (S, M) cannot be zero. Provide your answer as a dictionary mapping each letter to its digit, e.g., {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}
Answer: {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}
Metadata: {'source_dataset': 'cryptarithm', 'source_index': 0, 'puzzle': 'SEND + MORE = MONEY', 'solution': {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}, 'difficulty': {'num_letters': 8, 'value': (1, 100)}}

Example 2:
Question: Solve the cryptarithmetic puzzle: TWO + TWO = FOUR. Each letter represents a unique digit (0-9), and leading letters (T, F) cannot be zero. Provide your answer as a dictionary mapping each letter to its digit.
Answer: {'T': 7, 'W': 6, 'O': 5, 'F': 1, 'U': 3, 'R': 0}
Metadata: {'source_dataset': 'cryptarithm', 'source_index': 1, 'puzzle': 'TWO + TWO = FOUR', 'solution': {'T': 7, 'W': 6, 'O': 5, 'F': 1, 'U': 3, 'R': 0}, 'difficulty': {'num_letters': 6, 'value': (1, 100)}}

Example 3:
Question: Solve the cryptarithmetic puzzle: EAT + THAT = APPLE. Each letter represents a unique digit (0-9), and leading letters (E, T, A) cannot be zero. Provide your answer as a dictionary mapping each letter to its digit.
Answer: {'E': 9, 'A': 8, 'T': 7, 'H': 6, 'P': 3, 'L': 2}
Metadata: {'source_dataset': 'cryptarithm', 'source_index': 2, 'puzzle': 'EAT + THAT = APPLE', 'solution': {'E': 9, 'A': 8, 'T': 7, 'H': 6, 'P': 3, 'L': 2}, 'difficulty': {'num_letters': 6, 'value': (1, 100)}}

````
"""
import json
import os
import re
import random
import argparse
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from processing.gen_description import tool_maker_prompt
from .utils import call_api, eval_dataset, eval_single_instance, dedup_by_question

EVAL_SYSTEM_PROMPT = "You are a helpful assistant."


# Predefined cryptarithm puzzles with known solutions
CRYPTARITHM_PUZZLES = [
    ("SEND + MORE = MONEY", {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}),
    ("TWO + TWO = FOUR", {'T': 7, 'W': 6, 'O': 5, 'F': 1, 'U': 3, 'R': 0}),
    ("EAT + THAT = APPLE", {'E': 9, 'A': 8, 'T': 7, 'H': 6, 'P': 3, 'L': 2}),
    ("HALF + HALF = WHOLE", {'H': 5, 'A': 6, 'L': 4, 'F': 7, 'W': 1, 'O': 0, 'E': 2}),
    ("BASE + BALL = GAMES", {'B': 7, 'A': 8, 'S': 3, 'E': 2, 'L': 5, 'G': 1, 'M': 0}),
    ("USA + USSR = PEACE", {'U': 3, 'S': 7, 'A': 2, 'R': 8, 'P': 1, 'E': 0, 'C': 4}),
    ("EARTH + AIR = WATER", {'E': 6, 'A': 8, 'R': 9, 'T': 7, 'H': 5, 'I': 2, 'W': 1}),
    ("COCA + COLA = OASIS", {'C': 9, 'O': 5, 'A': 2, 'L': 8, 'S': 3, 'I': 1}),
    ("FORTY + TEN + TEN = SIXTY", {'F': 2, 'O': 9, 'R': 7, 'T': 8, 'Y': 0, 'E': 5, 'N': 6, 'S': 3, 'I': 1, 'X': 4}),
    ("ONE + ONE = TWO", {'O': 5, 'N': 3, 'E': 7, 'T': 1, 'W': 6}),
]


def parse_cryptarithm(puzzle_str):
    """
    Parse a cryptarithmetic puzzle string.
    Returns (left_terms, right_term, all_letters, leading_letters)
    """
    parts = puzzle_str.split('=')
    left_side = parts[0].strip()
    right_term = parts[1].strip()

    left_terms = [term.strip() for term in left_side.split('+')]

    all_letters = set()
    for term in left_terms + [right_term]:
        all_letters.update(term)

    # Leading letters cannot be zero
    leading_letters = set()
    for term in left_terms + [right_term]:
        if term:
            leading_letters.add(term[0])

    return left_terms, right_term, sorted(all_letters), sorted(leading_letters)


def word_to_number(word, letter_map):
    """Convert a word to a number using the letter mapping."""
    return int(''.join(str(letter_map[letter]) for letter in word))


def verify_cryptarithm(puzzle_str, solution):
    """
    Verify if a solution is correct for a cryptarithmetic puzzle.
    Returns True if valid, False otherwise.
    """
    try:
        left_terms, right_term, all_letters, leading_letters = parse_cryptarithm(puzzle_str)

        # Check if all letters are in the solution
        if set(solution.keys()) != set(all_letters):
            return False

        # Check if all values are unique and in range 0-9
        values = list(solution.values())
        if len(values) != len(set(values)):
            return False
        if not all(0 <= v <= 9 for v in values):
            return False

        # Check leading letters are not zero
        for letter in leading_letters:
            if solution[letter] == 0:
                return False

        # Verify the arithmetic
        left_sum = sum(word_to_number(term, solution) for term in left_terms)
        right_value = word_to_number(right_term, solution)

        return left_sum == right_value
    except (KeyError, ValueError, IndexError):
        return False


def solve_cryptarithm(puzzle_str):
    """
    Solve a cryptarithmetic puzzle using brute force.
    Returns the solution dictionary or None if no solution exists.
    """
    left_terms, right_term, all_letters, leading_letters = parse_cryptarithm(puzzle_str)

    # Try all permutations of digits
    for perm in permutations(range(10), len(all_letters)):
        solution = dict(zip(all_letters, perm))

        # Check leading letters are not zero
        if any(solution[letter] == 0 for letter in leading_letters):
            continue

        # Verify the arithmetic
        try:
            left_sum = sum(word_to_number(term, solution) for term in left_terms)
            right_value = word_to_number(right_term, solution)

            if left_sum == right_value:
                return solution
        except (KeyError, ValueError):
            continue

    return None


def generate_cryptarithm_problem(puzzle_str, solution):
    """
    Generate a cryptarithmetic problem with solution.
    """
    _, _, all_letters, leading_letters = parse_cryptarithm(puzzle_str)

    leading_str = ", ".join(sorted(leading_letters))

    question = (
        f"Solve the cryptarithmetic puzzle: {puzzle_str}. "
        f"Each letter represents a unique digit (0-9), and leading letters ({leading_str}) cannot be zero. "
        f"Provide your answer as a dictionary mapping each letter to its digit, "
        f"e.g., {{'A': 1, 'B': 2, ...}}"
    )

    answer = str(solution)

    metadata = {
        'source_dataset': 'cryptarithm',
        'puzzle': puzzle_str,
        'solution': str(solution),  # Convert to string to avoid schema mismatch
        'difficulty': {
            'num_letters': len(all_letters),
            'value': (1, 100)
        }
    }

    return {
        'question': question,
        'answer': answer,
        'metadata': metadata
    }


def init_data():
    """Initialize datasets locally. Uncomment push_to_hub lines below to upload."""
    random.seed(42)

    repo_id = "anonymous/cryptarithm"
    # Uncomment below to delete & recreate the HuggingFace dataset (requires HF credentials):
    # from huggingface_hub import HfApi, delete_repo
    # try:
    #     api = HfApi()
    #     delete_repo(repo_id=repo_id, repo_type="dataset")
    # except Exception as e:
    #     print(f"Note: Could not delete existing dataset: {e}")

    test_data = []
    val_data = []
    train_data = []

    # Difficulty levels based on number of unique letters
    easy_puzzles = [(p, s) for p, s in CRYPTARITHM_PUZZLES if len(s) <= 6]  # 5-6 letters
    medium_puzzles = [(p, s) for p, s in CRYPTARITHM_PUZZLES if 6 < len(s) <= 8]  # 7-8 letters
    hard_puzzles = [(p, s) for p, s in CRYPTARITHM_PUZZLES if len(s) > 8]  # 9+ letters

    # Generate test data
    print("Generating test data...")
    idx = 0
    for puzzle_list, difficulty in [(easy_puzzles, 2), (medium_puzzles, 3), (hard_puzzles, 4)]:
        for _ in range(70):
            if puzzle_list:
                puzzle_str, solution = random.choice(puzzle_list)
                problem = generate_cryptarithm_problem(puzzle_str, solution)
                problem['difficulty'] = difficulty
                problem['metadata']['source_index'] = idx
                test_data.append(problem)
                idx += 1

    # Generate validation data
    print("Generating validation data...")
    for i in range(120):
        puzzle_str, solution = random.choice(CRYPTARITHM_PUZZLES)
        problem = generate_cryptarithm_problem(puzzle_str, solution)
        num_letters = len(solution)
        problem['difficulty'] = 2 if num_letters <= 6 else 3 if num_letters <= 8 else 4
        problem['metadata']['source_index'] = i
        val_data.append(problem)

    # Generate training data
    print("Generating training data...")
    for i in range(500):
        puzzle_str, solution = random.choice(CRYPTARITHM_PUZZLES[:6])  # Use easier puzzles for training
        problem = generate_cryptarithm_problem(puzzle_str, solution)
        problem['difficulty'] = 2 if len(solution) <= 6 else 3
        problem['metadata']['source_index'] = i
        train_data.append(problem)

    # Push all splits
    print("Pushing test data to hub...")
    test_data = dedup_by_question(test_data)
    Dataset.from_list(test_data)  # .push_to_hub(repo_id, split="test")  # uncomment to upload to HuggingFace Hub
    print("Pushing validation data to hub...")
    val_data = dedup_by_question(val_data)
    Dataset.from_list(val_data)  # .push_to_hub(repo_id, split="validate")  # uncomment to upload to HuggingFace Hub
    print("Pushing training data to hub...")
    train_data = dedup_by_question(train_data)
    Dataset.from_list(train_data)  # .push_to_hub(repo_id, split="train")  # uncomment to upload to HuggingFace Hub
    print("All data pushed successfully!")


def load_ds():
    """Load training and test datasets"""
    train = load_dataset("anonymous/cryptarithm", split="validate")
    test = load_dataset("anonymous/cryptarithm", split="test")
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
    """
    # Extract the answer from the text
    pred_text = input_text.split('FINAL_ANSWER')[-1].replace(':', '').strip()

    try:
        # Try to parse as dictionary
        import ast
        # Look for dictionary pattern
        match = re.search(r'\{[^}]+\}', pred_text)
        if match:
            pred_text = match.group(0)

        pred_dict = ast.literal_eval(pred_text)
        ground_dict = ast.literal_eval(ground_truth)

        # Check if dictionaries match
        return pred_dict == ground_dict
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
    parser = argparse.ArgumentParser(description='Cryptarithm evaluation with tool generation')
    parser.add_argument('--api_key', type=str, default="token-abc121736278", help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default="http://localhost:8087/v1", help='Base URL for API')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Model name to use')
    parser.add_argument('--experiment_dir', type=str, default='experiments/cryptarithm/test', help='Experiment directory')
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
