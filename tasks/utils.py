import re
import json
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

EVAL_SYSTEM_PROMPT = "You are a helpful assistant"

def dedup_by_question(data):
    """Remove duplicate questions across difficulty levels, keeping the first occurrence."""
    seen = set()
    result = []
    for x in data:
        q = x['question']
        if q not in seen:
            seen.add(q)
            result.append(x)
    return result

def call_api(client, prompt, model_name, temperature=0.7, max_tokens=8192):
    """Call the appropriate API based on model name"""
    if model_name.startswith("claude") or model_name.startswith("anthropic"):
        # Anthropic API
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text, {}
    else:
        # OpenAI API
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content, {}

def load_tool_internally(tools, python_code_path):
    function_map = {}
    spec = importlib.util.spec_from_file_location("tool_module", python_code_path)
    tool_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tool_module) # TODO can we delete it afterwards?
    except (SyntaxError, ImportError) as e:
        return {}
    for tool_definition in tools:
        if 'function' in tool_definition:
            function_name = tool_definition['function']['name']
        else:
            function_name = tool_definition['name']
        function_to_call = getattr(tool_module, function_name)
        function_map[function_name] = function_to_call

    return function_map

def word_sorting_fn(final_answer_text, targets):
    json_code = re.findall(r'```json\n(.*?)\n```', final_answer_text, re.DOTALL)[-1]
    try:
        json_code = json.loads(json_code)
        targets = targets[0].split(' ')

        if len(json_code) != len(targets):
            return False

        for p, t in zip(json_code, targets):
            if p != t:
                return False
    except (json.JSONDecodeError, IndexError):
        return False

def eval_equality(input_text, ground_truth):
    pred_text = input_text.split('FINAL_ANSWER')[-1].replace(':', '').replace('*', '').split('\n')[0].strip()
    # print(pred_text, ground_truth)
    return ground_truth == pred_text


def eval_single_instance(row, tools, client, model_name, python_code_path=None, system_prompt=EVAL_SYSTEM_PROMPT, equal_fn = eval_equality):
    if python_code_path is None:
        function_map = {}
    else:
        function_map = load_tool_internally(tools, python_code_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row['inputs']}
    ]
    round_usage = {}
    final_answer_text = None

    # Initialize stats for this question
    stats = {
        'question': row['inputs'],
        'ground_truth': row['targets'],
        'total_rounds': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'tool_calls_made': 0,
        'tool_calls_successful': 0,
        'tool_calls_failed': 0,
        'final_answer_extracted': None,
        'correct': False,
        'used_tools': False,
        'conversation_length': 0,
        'tool_names_used': []
    }

    for rid in range(10):
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools, 
            # tool_choice= "required" if rid == 0 else "auto" 
            tool_choice= "auto"
        )
        round_usage[rid] = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'tool_used': 0,
        }

        # Update stats
        stats['total_rounds'] += 1
        stats['total_prompt_tokens'] += response.usage.prompt_tokens
        stats['total_completion_tokens'] += response.usage.completion_tokens

        first_call_time = time.time() - start_time
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            # Execute the tool call
            for tool_call in message.tool_calls:
                stats['tool_calls_made'] += 1
                function_args = json.loads(tool_call.function.arguments)
                name = tool_call.function.name

                if name not in function_map:
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": f"function : {name} does not exist"
                    })
                    stats['tool_calls_failed'] += 1
                else:
                    function_response = function_map[name](**function_args)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": str(function_response)
                    })
                    round_usage[rid]['tool_used'] += 1
                    stats['tool_calls_successful'] += 1
                    if name not in stats['tool_names_used']:
                        stats['tool_names_used'].append(name)
        else:
            messages.append(message)
            final_answer_text = message.content
            break

    # Final stats computation
    stats['used_tools'] = sum([r['tool_used'] for r in round_usage.values()]) > 0
    stats['conversation_length'] = len(messages)
    correct = equal_fn(final_answer_text, row['targets'])
    stats['correct'] = correct

    return correct, stats['used_tools'], messages, round_usage, stats

def eval_dataset(eval_ds, client, model_name, tools, python_code_path, log_results=False, system_prompt = EVAL_SYSTEM_PROMPT):
    # Dynamically import the Python function from the generated file
    print(system_prompt)
    spec = importlib.util.spec_from_file_location("tool_module", python_code_path)
    tool_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tool_module)
    except (SyntaxError, ImportError, AssertionError, IndexError, NameError, TypeError, ValueError) as e:
        return 0, 0
    valid_tools = []
    if log_results:
        log_file_jsonl = '/'.join(python_code_path.split('/')[:-1])+"+_evaluations_log.jsonl"
        print('logging final results')

    for tool_definition in tools:
        try:
            if 'function' in tool_definition:
                function_name = tool_definition['function']['name']
            else:
                function_name = tool_definition['name']
            getattr(tool_module, function_name)
            valid_tools.append(tool_definition)
        except (AttributeError, TypeError):
            continue

    if len(valid_tools) == 0:
        return 0, 0

    hit, total = 0, 0
    tool_usage = 0
    # Use ThreadPoolExecutor for parallel evaluation
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks to the thread pool
        future_to_row = {
            executor.submit(eval_single_instance, row, valid_tools, client, model_name, python_code_path, system_prompt): row
            for row in eval_ds
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(eval_ds), desc="Evaluating") as pbar:
            for future in as_completed(future_to_row):
                try:
                    correct, tool_used, messages, round_usage, stats = future.result()
                    tool_usage += tool_used
                    hit += correct
                    total += 1

                    # Log individual question results if requested
                    if log_results:
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

    return hit/total, tool_usage/total
