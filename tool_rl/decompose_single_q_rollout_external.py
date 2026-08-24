import ast
import json
import re
import time
from typing import Any

import openai

from tool_rl.tools import run_code
from tool_rl.decompose_rollout_utils import (
    _normalize_tool_schema as _normalize_tools,
    _extract_tool_names,
    _format_user_question,
)

EVALUATOR_BASE_URL = "http://localhost:9002/v1"
EVALUATOR_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
API_KEY = "TEST_API"

_EVALUATOR_CLIENT = openai.OpenAI(base_url=EVALUATOR_BASE_URL, api_key=API_KEY)

_BOX_RE = re.compile(r"\\box\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_ANSWER_RE = re.compile(r"FINAL[_\s]*ANSWER\s*:\s*([^\n\r]*)", re.IGNORECASE)


def _load_function_map_from_code(python_code: str, expected_names: list[str]) -> tuple[dict[str, Any], str | None]:
    """Extract function names from code via AST parsing — never executes the code.

    The function objects themselves are not needed: actual tool execution goes
    through _execute_tool_via_sandbox (subprocess + timeout).  We only need the
    set of defined names so we can check ``name not in function_map`` before
    dispatching.  Using exec() here caused infinite hangs when the model emitted
    top-level calls (e.g. a brute-force solver running at module scope).
    """
    if not python_code:
        return {}, "missing_python_code"

    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        return {}, f"code_load_error: {exc}"

    # Collect every function name defined anywhere in the code (top-level or
    # nested).  Using a sentinel value (True) because the map is only ever
    # tested for key membership; values are never invoked directly.
    defined_names: set[str] = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    function_map: dict[str, Any] = {}
    if expected_names:
        for fn_name in expected_names:
            if fn_name in defined_names:
                function_map[fn_name] = True  # sentinel; execution via _execute_tool_via_sandbox
    else:
        for name in defined_names:
            if not name.startswith("__"):
                function_map[name] = True
    return function_map, None


def _extract_final_answer_box_first(text: str | None) -> str | None:
    if not text:
        return None
    payload = text
    if "</think>" in payload:
        payload = payload.split("</think>")[-1]

    box_matches = list(_BOX_RE.finditer(payload))
    if box_matches:
        return box_matches[-1].group(1).strip()

    boxed_matches = list(_BOXED_RE.finditer(payload))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    final_matches = _FINAL_ANSWER_RE.findall(payload)
    if final_matches:
        candidate = final_matches[-1].strip()
        if candidate:
            return candidate
    return None


def _execute_tool_via_sandbox(
    *,
    python_code: str,
    function_name: str,
    function_args: dict[str, Any],
    tool_timeout: int,
) -> tuple[bool, str]:
    args_src = ", ".join(f"{k}={json.dumps(v)}" for k, v in (function_args or {}).items())
    script = (
        f"{python_code}\n\n"
        f"_result = {function_name}({args_src})\n"
        "print(_result)\n"
    )
    result = run_code(script, timeout=max(1, int(tool_timeout)))
    if result.get("status") == "Success":
        stdout = result.get("run_result", {}).get("stdout", "")
        return True, (stdout.strip() or "(no output)")
    stderr = result.get("run_result", {}).get("stderr", "Unknown error")
    return False, f"Error: {str(stderr)[:500]}"


def decompose_single_q_rollout_external(
    question: str,
    python_code: str,
    json_schema: str | dict | list[dict],
    max_rounds: int = 5,
    *,
    evaluator_client=None,
    evaluator_model: str | None = None,
    tool_timeout: int = 30,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    # Resolve at call time so that mutating EVALUATOR_MODEL_NAME after import takes effect.
    if evaluator_model is None:
        evaluator_model = EVALUATOR_MODEL_NAME
    client = evaluator_client or _EVALUATOR_CLIENT
    tools = _normalize_tools(json_schema)
    expected_tool_names = _extract_tool_names(tools)
    function_map, load_error = _load_function_map_from_code(python_code, expected_tool_names)

    messages: list[Any] = [
        {"role": "system", "content": "Use tools to solve the given question."},
        {"role": "user", "content": _format_user_question(question)},
    ]
    round_usage: dict[int, dict[str, int]] = {}
    final_answer_text = None

    stats = {
        "question": question,
        "total_rounds": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "tool_calls_made": 0,
        "tool_calls_successful": 0,
        "tool_calls_failed": 0,
        "tool_names_used": [],
        "used_tools": False,
        "code_load_error": load_error,
    }

    if load_error:
        return {
            "ok": False,
            "error": load_error,
            "messages": messages,
            "round_usage": round_usage,
            "final_answer_text": None,
            "extracted_answer": None,
            "stats": stats,
            "used_tools": False,
        }

    for rid in range(max_rounds):
        start_time = time.time()
        # Force tool use on the first round so the evaluator cannot skip tools and
        # answer directly with plain text (tool_choice="auto" lets the model do that,
        # causing tool_calls_successful=0 → build_eval_reward=0 silently).
        # From round 1 onward, revert to the caller-supplied tool_choice ("auto") so
        # the model is free to give a final answer after receiving the tool result.
        effective_tool_choice = "required" if (rid == 0 and tools) else tool_choice
        try:
            if tools:
                response = client.chat.completions.create(
                    model=evaluator_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=effective_tool_choice,
                    temperature=0.6,
                    max_tokens=1024,
                    timeout=tool_timeout,
                )
            else:
                response = client.chat.completions.create(
                    model=evaluator_model,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=1024,
                    timeout=tool_timeout,
                )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"evaluator_api_error: {exc}",
                "messages": messages,
                "round_usage": round_usage,
                "final_answer_text": final_answer_text,
                "extracted_answer": _extract_final_answer_box_first(final_answer_text),
                "stats": stats,
                "used_tools": bool(stats["tool_calls_successful"]),
            }

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        round_usage[rid] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_used": 0,
        }

        stats["total_rounds"] += 1
        stats["total_prompt_tokens"] += prompt_tokens
        stats["total_completion_tokens"] += completion_tokens
        _ = time.time() - start_time

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            messages.append(message)
            for tool_call in tool_calls:
                stats["tool_calls_made"] += 1
                name = getattr(tool_call.function, "name", "")
                raw_args = getattr(tool_call.function, "arguments", "{}")
                try:
                    function_args = json.loads(raw_args or "{}")
                    if not isinstance(function_args, dict):
                        function_args = {}
                except json.JSONDecodeError:
                    function_args = {}

                if name not in function_map:
                    tool_content = f"function : {name} does not exist"
                    stats["tool_calls_failed"] += 1
                else:
                    ok, tool_content = _execute_tool_via_sandbox(
                        python_code=python_code,
                        function_name=name,
                        function_args=function_args,
                        tool_timeout=tool_timeout,
                    )
                    if ok:
                        round_usage[rid]["tool_used"] += 1
                        stats["tool_calls_successful"] += 1
                        if name not in stats["tool_names_used"]:
                            stats["tool_names_used"].append(name)
                    else:
                        stats["tool_calls_failed"] += 1

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": name,
                        "content": str(tool_content),
                    }
                )
        else:
            messages.append(message)
            final_answer_text = message.content or ""
            break

    extracted_answer = _extract_final_answer_box_first(final_answer_text)
    stats["used_tools"] = bool(stats["tool_calls_successful"])

    return {
        "ok": True,
        "error": None,
        "messages": messages,
        "round_usage": round_usage,
        "final_answer_text": final_answer_text,
        "extracted_answer": extracted_answer,
        "stats": stats,
        "used_tools": stats["used_tools"],
    }


if __name__ == "__main__":
    _TEST_PYTHON_CODE = """
def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b
"""

    _TEST_JSON_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]

    _TEST_QUESTION = "What is (3 + 7) * 5?"

    print(f"Testing decompose_single_q_rollout_external")
    print(f"  URL   : {EVALUATOR_BASE_URL}")
    print(f"  Model : {EVALUATOR_MODEL_NAME}")
    print(f"  Q     : {_TEST_QUESTION}")
    print()

    result = decompose_single_q_rollout_external(
        question=_TEST_QUESTION,
        python_code=_TEST_PYTHON_CODE,
        json_schema=_TEST_JSON_SCHEMA,
        max_rounds=5,
    )

    print(f"ok              : {result['ok']}")
    print(f"error           : {result['error']}")
    print(f"extracted_answer: {result['extracted_answer']}")
    print(f"used_tools      : {result['used_tools']}")
    print()
    print("--- stats ---")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
    print()
    print("--- round_usage ---")
    for rid, usage in result["round_usage"].items():
        print(f"  round {rid}: {usage}")
    print()
    print("--- messages ---")
    for msg in result["messages"]:
        role = getattr(msg, "role", None) or msg.get("role", "?")
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            calls_summary = [f"{tc.function.name}({tc.function.arguments})" for tc in tool_calls]
            print(f"  [{role}] <tool_calls> {calls_summary}")
        else:
            snippet = str(content)[:120].replace("\n", " ")
            print(f"  [{role}] {snippet}")