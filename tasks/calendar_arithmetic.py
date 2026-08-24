"""
### calendar_arithmetic
Generates calendar arithmetic puzzle tasks

Calendar arithmetic puzzles involve date calculations such as:
- Finding day of the week for a given date
- Calculating days between dates
- Adding/subtracting days from dates
- Finding dates that satisfy certain conditions

Default configuration:
```python
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: What day of the week was January 1, 2000?
Answer: Saturday
Metadata: {'source_dataset': 'calendar_arithmetic', 'source_index': 0, 'task_type': 'day_of_week', 'date': '2000-01-01', 'answer': 'Saturday', 'difficulty': {'value': (1, 100)}}

Example 2:
Question: How many days are there between March 15, 2020 and June 22, 2020 (inclusive)?
Answer: 100
Metadata: {'source_dataset': 'calendar_arithmetic', 'source_index': 1, 'task_type': 'days_between', 'start_date': '2020-03-15', 'end_date': '2020-06-22', 'answer': 100, 'difficulty': {'value': (1, 100)}}

Example 3:
Question: What date is 45 days after February 10, 2024?
Answer: 2024-03-26
Metadata: {'source_dataset': 'calendar_arithmetic', 'source_index': 2, 'task_type': 'date_add', 'start_date': '2024-02-10', 'days_to_add': 45, 'answer': '2024-03-26', 'difficulty': {'value': (1, 100)}}

````
"""
import json
import os
import re
import random
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from processing.gen_description import tool_maker_prompt
from .utils import call_api, eval_dataset, eval_single_instance, dedup_by_question

EVAL_SYSTEM_PROMPT = "You are a helpful assistant."


def is_leap_year(year):
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def get_day_of_week(date):
    """Get day of the week for a given date."""
    return date.strftime('%A')


def days_between(start_date, end_date, inclusive=True):
    """Calculate days between two dates."""
    delta = (end_date - start_date).days
    return delta + 1 if inclusive else delta


def add_days(start_date, days):
    """Add days to a date."""
    return start_date + timedelta(days=days)


def subtract_days(start_date, days):
    """Subtract days from a date."""
    return start_date - timedelta(days=days)


def generate_day_of_week_problem():
    """Generate a day of the week problem."""
    year = random.randint(1900, 2100)
    month = random.randint(1, 12)

    # Get max day for the month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:  # February
        max_day = 29 if is_leap_year(year) else 28

    day = random.randint(1, max_day)
    date = datetime(year, month, day)

    day_name = get_day_of_week(date)
    date_str = date.strftime('%B %d, %Y')

    question = f"What day of the week was {date_str}?"
    answer = day_name

    metadata = {
        'source_dataset': 'calendar_arithmetic',
        'task_type': 'day_of_week',
        'date': date.strftime('%Y-%m-%d'),
        'answer': day_name,
        'difficulty': {'value': (1, 100)}
    }

    return {'question': question, 'answer': answer, 'metadata': metadata}


def generate_days_between_problem():
    """Generate a days between problem."""
    year = random.randint(2000, 2030)
    start_month = random.randint(1, 12)
    start_day = random.randint(1, 28)  # Safe day for all months

    start_date = datetime(year, start_month, start_day)
    days_delta = random.randint(30, 365)
    end_date = start_date + timedelta(days=days_delta - 1)

    start_str = start_date.strftime('%B %d, %Y')
    end_str = end_date.strftime('%B %d, %Y')

    question = f"How many days are there between {start_str} and {end_str} (inclusive)?"
    answer = str(days_delta)

    metadata = {
        'source_dataset': 'calendar_arithmetic',
        'task_type': 'days_between',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'answer': str(days_delta),
        'difficulty': {'value': (1, 100)}
    }

    return {'question': question, 'answer': answer, 'metadata': metadata}


def generate_date_add_problem():
    """Generate a date addition problem."""
    year = random.randint(2000, 2030)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    start_date = datetime(year, month, day)
    days_to_add = random.randint(1, 200)
    end_date = start_date + timedelta(days=days_to_add)

    start_str = start_date.strftime('%B %d, %Y')

    question = f"What date is {days_to_add} days after {start_str}?"
    answer = end_date.strftime('%Y-%m-%d')

    metadata = {
        'source_dataset': 'calendar_arithmetic',
        'task_type': 'date_add',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'days_to_add': days_to_add,
        'answer': answer,
        'difficulty': {'value': (1, 100)}
    }

    return {'question': question, 'answer': answer, 'metadata': metadata}


def generate_date_subtract_problem():
    """Generate a date subtraction problem."""
    year = random.randint(2000, 2030)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    start_date = datetime(year, month, day)
    days_to_subtract = random.randint(1, 200)
    end_date = start_date - timedelta(days=days_to_subtract)

    start_str = start_date.strftime('%B %d, %Y')

    question = f"What date is {days_to_subtract} days before {start_str}?"
    answer = end_date.strftime('%Y-%m-%d')

    metadata = {
        'source_dataset': 'calendar_arithmetic',
        'task_type': 'date_subtract',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'days_to_subtract': days_to_subtract,
        'answer': answer,
        'difficulty': {'value': (1, 100)}
    }

    return {'question': question, 'answer': answer, 'metadata': metadata}


def generate_leap_year_problem():
    """Generate a leap year problem."""
    year = random.randint(1900, 2100)
    is_leap = is_leap_year(year)

    question = f"Is {year} a leap year? Answer 'Yes' or 'No'."
    answer = "Yes" if is_leap else "No"

    metadata = {
        'source_dataset': 'calendar_arithmetic',
        'task_type': 'leap_year',
        'year': year,
        'answer': answer,
        'difficulty': {'value': (1, 100)}
    }

    return {'question': question, 'answer': answer, 'metadata': metadata}


PROBLEM_GENERATORS = [
    generate_day_of_week_problem,
    generate_days_between_problem,
    generate_date_add_problem,
    generate_date_subtract_problem,
    generate_leap_year_problem,
]


def init_data():
    """Initialize and push datasets to HuggingFace Hub"""
    random.seed(42)

    test_data = []
    val_data = []
    train_data = []

    # Generate test data - mix of all problem types
    for i in range(210):
        generator = random.choice(PROBLEM_GENERATORS)
        problem = generator()
        problem['difficulty'] = 2 + (i // 70)  # Difficulty 2, 3, 4
        problem['metadata']['source_index'] = i
        test_data.append(problem)

    # Generate validation data
    for i in range(120):
        generator = random.choice(PROBLEM_GENERATORS)
        problem = generator()
        problem['difficulty'] = 2
        problem['metadata']['source_index'] = i
        val_data.append(problem)

    # Generate training data
    for i in range(500):
        generator = random.choice(PROBLEM_GENERATORS)
        problem = generator()
        problem['difficulty'] = 2
        problem['metadata']['source_index'] = i
        train_data.append(problem)

    test_data = dedup_by_question(test_data)
    Dataset.from_list(test_data)  # .push_to_hub("anonymous/calendar_arithmetic", split="test")  # uncomment to upload to HuggingFace Hub
    val_data = dedup_by_question(val_data)
    Dataset.from_list(val_data)  # .push_to_hub("anonymous/calendar_arithmetic", split="validate")  # uncomment to upload to HuggingFace Hub
    train_data = dedup_by_question(train_data)
    Dataset.from_list(train_data)  # .push_to_hub("anonymous/calendar_arithmetic", split="train")  # uncomment to upload to HuggingFace Hub


def load_ds():
    """Load training and test datasets"""
    train = load_dataset("anonymous/calendar_arithmetic", split="validate")
    test = load_dataset("anonymous/calendar_arithmetic", split="test")
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

    # Remove common formatting
    pred_text = pred_text.split('\n')[0].strip()

    # Try exact match first
    if pred_text == ground_truth:
        return True

    # Try case-insensitive match for day names and Yes/No
    if pred_text.lower() == ground_truth.lower():
        return True

    # Try to extract numbers for numeric answers
    try:
        pred_num = int(re.search(r'\d+', pred_text).group())
        ground_num = int(ground_truth)
        return pred_num == ground_num
    except (AttributeError, ValueError):
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
    parser = argparse.ArgumentParser(description='Calendar arithmetic evaluation with tool generation')
    parser.add_argument('--api_key', type=str, default="token-abc121736278", help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default="http://localhost:8087/v1", help='Base URL for API')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Model name to use')
    parser.add_argument('--experiment_dir', type=str, default='experiments/calendar_arithmetic/test', help='Experiment directory')
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
