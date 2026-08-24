"""
Shared utility functions for decomposed rollout implementations.

Extracted from decompose_rollout_manual_refact_external_llm.py,
decompose_rollout_manual_refact_w_judge_rewards.py, and
decompose_rollout_manual_refact_w_judge_rewards_with_tool_pool.py.
"""
import ast
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from tool_rl.tools import run_code

_METADATA_RE = re.compile(r"<tool_rl_metadata>\s*(.*?)\s*</tool_rl_metadata>", re.DOTALL)
_MAIN_GUARD_RE = re.compile(
    r'^\s*if\s+(?:__name__\s*==\s*([\'\"])__main__\1|([\'\"])__main__\2\s*==\s*__name__)\s*:\s*(?:#.*)?$'
)
TOOL_EXEC_SUCCESS_REWARD = 0.3
_FINAL_ANSWER_PROMPT_PREFIX = (
    "Solve the following question. When you have the final answer, present it as:\n"
    "\\boxed{your answer here}\n"
)
_FINAL_ANSWER_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _format_user_question(question: str) -> str:
    return f"{_FINAL_ANSWER_PROMPT_PREFIX}{question}"


def _extract_final_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{...} in model output."""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def _strip_metadata(prompt_text: str) -> tuple[dict, str]:
    """Extract metadata JSON and remove it from prompt text."""
    if not prompt_text:
        return {}, prompt_text

    match = _METADATA_RE.search(prompt_text)
    if not match:
        return {}, prompt_text

    meta_str = match.group(1).strip()
    metadata = {}
    if meta_str:
        try:
            metadata = json.loads(meta_str)
        except json.JSONDecodeError:
            metadata = {}

    cleaned = prompt_text[:match.start()] + prompt_text[match.end():]
    return metadata, cleaned


def extract_code_blocks(text: str, language: str) -> list[str]:
    """Extract all code blocks for a given language from markdown fences."""
    pattern = rf"```{language}\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def _apply_chat_template(tokenizer, messages, tools=None, add_generation_prompt=True) -> str:
    """Apply chat template with best-effort support for tools + generation prompt."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)


def _parse_chatml_messages(prompt_text: str) -> list[dict] | None:
    """Parse ChatML-like prompt text into messages (best effort)."""
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt_text, re.DOTALL)
    if not matches:
        return None
    return [{"role": role, "content": content} for role, content in matches]


def _normalize_tool_schema(raw_schema: dict) -> list[dict]:
    """Normalize tool schema to OpenAI tools list."""
    if "functions" in raw_schema:
        tool_schema = raw_schema["functions"]
    elif "tools" in raw_schema:
        tool_schema = raw_schema["tools"]
    else:
        tool_schema = raw_schema

    if isinstance(tool_schema, dict):
        tool_schema = [tool_schema]

    openai_tools = []
    for tool in tool_schema:
        if isinstance(tool, dict) and "function" in tool:
            openai_tools.append(tool)
        elif isinstance(tool, dict):
            openai_tools.append({"type": "function", "function": tool})
    return openai_tools


def _extract_tool_names(openai_tools: list[dict]) -> list[str]:
    if not openai_tools:
        return []
    names: list[str] = []
    for tool in openai_tools:
        if not isinstance(tool, dict):
            continue
        tool_fn = tool.get("function", {})
        if isinstance(tool_fn, dict):
            name = tool_fn.get("name")
            if name:
                names.append(name)
    return names


def _extract_sampled_logprobs(output_choice) -> list[float] | None:
    """Extract sampled-token logprobs from one vLLM output choice."""
    if not output_choice.logprobs:
        return None

    values: list[float] = []
    for token_logprobs_dict in output_choice.logprobs:
        if token_logprobs_dict:
            sampled_logprob = next(iter(token_logprobs_dict.values()))
            values.append(sampled_logprob.logprob if hasattr(sampled_logprob, "logprob") else 0.0)
        else:
            values.append(0.0)
    return values


def _remove_dunder_main_block(code: str | None) -> str | None:
    """Strip top-level `if __name__ == "__main__":` blocks from generated tool code."""
    if not code:
        return code

    lines = code.splitlines()
    output_lines: list[str] = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        if not _MAIN_GUARD_RE.match(line):
            output_lines.append(line)
            i += 1
            continue

        changed = True
        guard_indent = len(line) - len(line.lstrip(" "))
        i += 1

        while i < len(lines):
            next_line = lines[i]
            stripped = next_line.strip()
            if not stripped:
                i += 1
                continue

            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= guard_indent:
                break
            i += 1

    if not changed:
        return code
    return "\n".join(output_lines).strip()


def _extract_function_names_from_code(code: str | None) -> list[str]:
    if not code:
        return []
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
    return names


def _extract_function_params_from_code(code: str | None) -> list[str]:
    if not code:
        return []
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef):
            return [arg.arg for arg in node.args.args]
    return []


def _extract_function_params_map(code: str | None) -> dict[str, list[str]]:
    if not code:
        return {}
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return {}
    params_map: dict[str, list[str]] = {}
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef):
            params_map[node.name] = [arg.arg for arg in node.args.args]
    return params_map


def _extract_param_names_from_schema(openai_tools: list[dict]) -> list[str]:
    if not openai_tools:
        return []
    tool_fn = openai_tools[0].get("function", {}) if isinstance(openai_tools[0], dict) else {}
    params = tool_fn.get("parameters", {}) if isinstance(tool_fn, dict) else {}
    if not isinstance(params, dict):
        return []
    required = params.get("required")
    if isinstance(required, list) and required:
        return [name for name in required if isinstance(name, str) and name]
    props = params.get("properties", {})
    if isinstance(props, dict) and props:
        return [name for name in props.keys() if isinstance(name, str) and name]
    return []


def _remap_call_args(call_args: dict, expected_param_names: list[str] | None) -> dict:
    """Remap model's argument keys to the function's actual parameter names.

    Small models often use generic keys (e.g. ``question``, ``problem_text``)
    instead of the schema's actual parameter name.  When the number of values
    matches the number of expected parameters we re-key accordingly; when there
    is a single value and a single expected param we always remap.
    """
    if not expected_param_names or not isinstance(call_args, dict) or not call_args:
        return call_args

    model_keys = list(call_args.keys())

    # Already correct – every model key is a known param name
    if all(k in expected_param_names for k in model_keys):
        return call_args

    values = list(call_args.values())

    # Single-param shortcut (most common case)
    if len(values) == 1 and len(expected_param_names) >= 1:
        return {expected_param_names[0]: values[0]}

    # Multi-param: positional remap when counts match
    if len(values) == len(expected_param_names):
        return dict(zip(expected_param_names, values))

    # Fallback: keep original (may fail, but avoids silent data loss)
    return call_args


def _coerce_call_args(raw_args: Any) -> dict:
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def _filter_valid_tool_calls(
    tool_calls: list[dict],
    allowed_tool_names: set[str] | None,
    fallback_tool_name: str | None,
) -> list[dict]:
    valid_calls: list[dict] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        if allowed_tool_names:
            if tc.get("name") in allowed_tool_names:
                valid_calls.append(tc)
        elif fallback_tool_name and tc.get("name") == fallback_tool_name:
            valid_calls.append(tc)
    return valid_calls


def extract_tool_calls_from_text(text: str) -> list[dict]:
    """Extract all pseudo-tool-calls from model output."""
    calls = []
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    for m in matches:
        try:
            calls.append(json.loads(m))
        except json.JSONDecodeError:
            continue
    return calls


def _execute_tool_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call; designed for optional threadpool dispatch."""
    use_code = _remove_dunder_main_block(job.get("use_code", "")) or ""
    use_func = job.get("use_func")
    call_args = job.get("call_args", {}) or {}
    expected_param_names = job.get("expected_param_names")
    timeout = int(job.get("timeout", 10))

    call_args = _remap_call_args(call_args, expected_param_names)

    if use_code and use_func:
        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in call_args.items())
        script = f"""{use_code}\n\n_result = {use_func}({args_str})\nprint(_result)\n"""
        result = run_code(script, timeout=timeout)
        if result.get("status") == "Success":
            tool_result = result["run_result"]["stdout"].strip() or "(no output)"
            reward_delta = TOOL_EXEC_SUCCESS_REWARD
        else:
            stderr = result.get("run_result", {}).get("stderr", "Unknown error")
            tool_result = f"Error: {stderr[:300]}"
            reward_delta = -0.2
    else:
        tool_result = "Error: Missing code or function_name"
        reward_delta = -0.2

    return {
        "state_idx": job["state_idx"],
        "step_text": job["step_text"],
        "tool_result": tool_result,
        "reward_delta": reward_delta,
        "tool_name": job.get("tool_name"),
    }


def _safe_preview(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    return text[:limit]


def _format_messages_for_log(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _maybe_print_messages(trainer, state: dict, header: str):
    if not getattr(trainer, "log_rollout_messages", False):
        return
    messages = state.get("messages")
    if not messages:
        return
    python_code = state.get("python_code")
    task_type = state.get("task_type", "unknown")
    rollout_id = state.get("rollout_session_id", "unknown")
    prompt_index = state.get("prompt_index", -1)
    print(f"\n===== {header} | task={task_type} | rollout={rollout_id} | prompt_index={prompt_index} =====")
    if python_code:
        print("----- TOOL PYTHON CODE -----")
        print(python_code)
        print("----- END TOOL PYTHON CODE -----\n")
    print(_format_messages_for_log(messages))
    print("===== END ROLLOUT MESSAGES =====\n")


def _get_rollout_logger(trainer):
    """Build a lightweight JSONL logger for custom rollout events."""
    log_path = getattr(trainer, "rollout_log_path", None)
    if not log_path:
        return None

    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    except OSError:
        return None

    accelerator = getattr(trainer, "accelerator", None)
    rank = getattr(accelerator, "process_index", 0)
    world_size = getattr(accelerator, "num_processes", 1)
    pid = os.getpid()

    def _log_event(event_type: str, **payload):
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "rank": rank,
            "world_size": world_size,
            "pid": pid,
        }
        event.update(payload)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    return _log_event


def _get_llm_engine(trainer):
    """Return the vLLM engine handle from trainer (colocate mode)."""
    if hasattr(trainer, "llm") and trainer.llm is not None:
        return trainer.llm
    vllm_gen = getattr(trainer, "vllm_generation", None)
    if vllm_gen is not None and getattr(vllm_gen, "llm", None) is not None:
        return vllm_gen.llm
    raise AttributeError(
        "vLLM engine not found on trainer. Use vLLM colocate mode or set trainer.llm."
    )


def _has_exactly_one_python_and_one_json_block(text: str) -> bool:
    """Check strict Step-1 format: exactly one python block and one json block."""
    fenced_langs = [
        lang.strip().lower()
        for lang in re.findall(r"```([^\n`]*)\n.*?\n```", text, re.DOTALL)
    ]
    return len(fenced_langs) == 2 and fenced_langs.count("python") == 1 and fenced_langs.count("json") == 1
