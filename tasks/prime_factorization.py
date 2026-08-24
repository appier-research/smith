""" ### prime_factorization
Generates prime factorization tasks

Default configuration:
```python
min_value = 2
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the prime factorization of 656. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 2 × 2 × 41
Metadata: {'source_dataset': 'prime_factorization', 'source_index': 0, 'number': 656, 'factors': [2, 2, 2, 2, 41], 'difficulty': {'value': (2, 1000)}}

Example 2:
Question: Find the prime factorization of 41. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 41
Metadata: {'source_dataset': 'prime_factorization', 'source_index': 1, 'number': 41, 'factors': [41], 'difficulty': {'value': (2, 1000)}}

Example 3:
Question: Find the prime factorization of 420. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 3 × 5 × 7
Metadata: {'source_dataset': 'prime_factorization', 'source_index': 2, 'number': 420, 'factors': [2, 2, 3, 5, 7], 'difficulty': {'value': (2, 1000)}}
````
"""
import json
import os
import re
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from processing.gen_description import tool_maker_prompt
from .utils import call_api, eval_dataset, eval_single_instance, dedup_by_question

CUSTOM_EVAL_SYSTEM_PROMPT = "You are a helpful assistant. Make sure to keep your final answer at FINAL_ANSWER: . Do not add any markdown format"

def init_data():
    import reasoning_gym
    from datasets import Dataset
    test_data = []
    data = reasoning_gym.create_dataset('prime_factorization', size=70, seed=42, min_value=2, max_value=500)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 2
        test_data.append(x)

    data = reasoning_gym.create_dataset('prime_factorization', size=70, seed=42, min_value=500, max_value=2000)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 3
        test_data.append(x)

    data = reasoning_gym.create_dataset('prime_factorization', size=70, seed=42, min_value=2000, max_value=5000)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 4
        test_data.append(x)
    test_data = dedup_by_question(test_data)
    Dataset.from_list(test_data, split="test")  # .push_to_hub("anonymous/prime_factorization")  # uncomment to upload to HuggingFace Hub

    val_data = []
    data = reasoning_gym.create_dataset('prime_factorization', size=120, seed=42, min_value=2, max_value=1000)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 2
        val_data.append(x)
    val_data = dedup_by_question(val_data)
    Dataset.from_list(val_data, split="validate")  # .push_to_hub("anonymous/prime_factorization")  # uncomment to upload to HuggingFace Hub

    train_data = []
    data = reasoning_gym.create_dataset('prime_factorization', size=500, seed=42, min_value=2, max_value=500)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 2
        train_data.append(x)

    data = reasoning_gym.create_dataset('prime_factorization', size=500, seed=42, min_value=500, max_value=2000)
    for i, x in enumerate(data):
        print('metadata:', x['metadata'])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x['answer'], entry=x) == 1.0
        x['difficulty'] = 3
        train_data.append(x)
    train_data = dedup_by_question(train_data)
    Dataset.from_list(train_data, split="train")  # .push_to_hub("anonymous/prime_factorization")  # uncomment to upload to HuggingFace Hub

def load_ds():
    train = load_dataset("anonymous/prime_factorization", split="validate")
    test = load_dataset("anonymous/prime_factorization", split="test")
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
    pred_text = input_text.split('FINAL_ANSWER')[-1].replace(':', '').replace('*', '').split('\n')[0].strip()
    return ground_truth == pred_text


def baseline(eval_ds, client, model_name, system_prompt, experiment_dir=None):
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
            executor.submit(eval_single_instance, row, [], client, model_name, None, CUSTOM_EVAL_SYSTEM_PROMPT, eval_equality): row
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
        acc, tool_usage_rate = eval_dataset(test_q, client, model_name, openai_tools, os.path.join(output_dir, f"function.py"), False, CUSTOM_EVAL_SYSTEM_PROMPT)
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
    acc, tool_usage_rate = eval_dataset(test, client, model_name, openai_tools, function_py_path, log_results=True, system_prompt=CUSTOM_EVAL_SYSTEM_PROMPT)
    print(f"Final test accuracy: {acc:.3f} ({tool_usage_rate:.3f})")

    return acc


if __name__ == "__main__":
    # init_data()
    parser = argparse.ArgumentParser(description='Prime Factorization evaluation with tool generation')
    parser.add_argument('--api_key', type=str, default="token-abc121736278", help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default="http://localhost:8087/v1", help='Base URL for API')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Model name to use')
    parser.add_argument('--experiment_dir', type=str, default='experiments/prime_factorization/test', help='Experiment directory')
    parser.add_argument('--baseline', action='store_true', help='Run baseline evaluation without tool generation')
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    train, test = load_ds()

    if args.baseline:
        # Run baseline evaluation
        print(f"\nTest set size: {len(test)}")
        test_acc = baseline(test, client, args.model_name, CUSTOM_EVAL_SYSTEM_PROMPT, args.experiment_dir)
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
