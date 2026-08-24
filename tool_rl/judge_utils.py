"""
Shared LLM-judge utilities for tool quality scoring.

Used by:
  - dataset_creation/judge_tool_quality.py  (offline batch judging)
  - tool_rl/decompose_rollout_manual_refact_w_judge_rewards.py  (online rollout judging)
"""
import ast
import json
import logging
import os
import re
import time

import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt (loaded once at import time)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPT_PATH = os.path.join(_HERE, "..", "dataset_creation", "judge_prompts", "prompt_v5.txt")

try:
    with open(_PROMPT_PATH, "r") as _f:
        JUDGE_SYSTEM_PROMPT: str = _f.read()
except FileNotFoundError:
    logger.warning("Judge system prompt not found at %s; judge calls will fail.", _PROMPT_PATH)
    JUDGE_SYSTEM_PROMPT = ""

SCORE_DIMS = [
    "code_correctness",
    "code_clarity",
    "schema_quality",
    "schema_code_alignment",
    "overall_quality",
]

MAX_CODE_CHARS = 6_000


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_json_scores(raw: str) -> dict:
    """Extract a JSON object with quality scores from raw LLM output."""
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    m = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m2.group(0).strip() if m2 else raw.strip()

    parsed = json.loads(candidate)

    for dim in SCORE_DIMS:
        if dim not in parsed:
            raise ValueError(f"Missing dimension: {dim}")
        val = parsed[dim]
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid score for {dim}: {val!r}")
        if not (0 <= val <= 5):
            raise ValueError(f"Score out of range for {dim}: {val!r}")
        parsed[dim] = val

    return parsed


# ---------------------------------------------------------------------------
# Schema ↔ code structural alignment check
# ---------------------------------------------------------------------------

def check_schema_code_alignment(python_code: str, openai_tools: list[dict]) -> tuple[bool, str]:
    """Programmatically verify that the named tool in the schema has a matching
    callable in the Python code with the same parameter names.

    Handles two callable kinds:
    - Top-level ``def <name>(...)``: checks argument names directly.
    - ``class <name>``: checks ``__init__`` or ``__call__`` argument names
      (excluding ``self``).

    Returns:
        (is_aligned, reason) – ``True`` if the schema and code agree, else
        ``False`` with a human-readable reason string.
    """
    if not openai_tools or not python_code:
        return False, "missing code or schema"

    # --- extract schema name + param names ---
    raw = openai_tools[0]
    fn_schema = raw.get("function", raw) if isinstance(raw, dict) else {}
    tool_name: str = fn_schema.get("name", "")
    if not tool_name:
        return False, "schema has no 'name' field"

    props = fn_schema.get("parameters", {}).get("properties", {})
    required = fn_schema.get("parameters", {}).get("required", [])
    # prefer required list; fall back to all properties
    schema_params: set[str] = set(required) if required else set(props.keys())

    # --- parse Python AST ---
    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        return False, f"syntax error in code: {exc}"

    # Build a flat map: callable_name → [param, ...]
    callables: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg != "self"]
            callables[node.name] = params
        elif isinstance(node, ast.ClassDef):
            # look for __init__ or __call__ inside the class body
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in (
                    "__init__",
                    "__call__",
                ):
                    params = [a.arg for a in item.args.args if a.arg != "self"]
                    callables[node.name] = params
                    break
            # also register the class name without params if no __init__/__call__
            if node.name not in callables:
                callables[node.name] = []

    if tool_name not in callables:
        return False, (
            f"callable '{tool_name}' not found in code "
            f"(found: {sorted(callables.keys())})"
        )

    code_params: set[str] = set(callables[tool_name])
    if schema_params != code_params:
        return False, (
            f"parameter mismatch for '{tool_name}': "
            f"schema={sorted(schema_params)}, code={sorted(code_params)}"
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# OpenAI tool schema structural validity check
# ---------------------------------------------------------------------------

_VALID_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def validate_openai_tool_schema(openai_tools: list[dict]) -> tuple[bool, str]:
    """Check that *openai_tools* is a structurally valid OpenAI function-calling
    schema list.  This is a lightweight structural check — not a full JSON Schema
    validator — but it catches the common failure modes that cause the evaluator
    API to reject or silently mishandle the schema:

    Per-tool checks:
      1. Each entry is a dict.
      2. ``function`` key is present and is a dict.
      3. ``function.name`` is a non-empty string.
      4. ``function.description`` is present and is a string (may be empty).
      5. ``function.parameters`` is present and is a dict.
      6. ``parameters.type`` equals ``"object"``.
      7. ``parameters.properties`` is present and is a dict.
      8. Each property value is a dict containing a ``"type"`` key whose value
         is one of the standard JSON Schema primitive types.
      9. ``parameters.required`` (if present) is a list of strings that are all
         keys in ``parameters.properties``.

    Returns:
        (is_valid, reason) – ``True`` if the schema passes all checks, else
        ``False`` with a human-readable description of the first failure found.
    """
    if not openai_tools:
        return False, "openai_tools is empty"

    for i, tool in enumerate(openai_tools):
        prefix = f"tool[{i}]"

        if not isinstance(tool, dict):
            return False, f"{prefix}: expected dict, got {type(tool).__name__}"

        fn = tool.get("function")
        if not isinstance(fn, dict):
            return False, f"{prefix}: 'function' key missing or not a dict"

        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return False, f"{prefix}: 'function.name' missing or empty"

        if "description" not in fn:
            return False, f"{prefix}({name}): 'function.description' missing"
        if not isinstance(fn["description"], str):
            return False, f"{prefix}({name}): 'function.description' must be a string"

        params = fn.get("parameters")
        if not isinstance(params, dict):
            return False, f"{prefix}({name}): 'function.parameters' missing or not a dict"

        if params.get("type") != "object":
            return False, (
                f"{prefix}({name}): 'parameters.type' must be 'object', "
                f"got {params.get('type')!r}"
            )

        props = params.get("properties")
        if not isinstance(props, dict):
            return False, f"{prefix}({name}): 'parameters.properties' missing or not a dict"

        for prop_name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                return False, (
                    f"{prefix}({name}): property '{prop_name}' schema must be a dict, "
                    f"got {type(prop_schema).__name__}"
                )
            prop_type = prop_schema.get("type")
            if prop_type is None:
                return False, f"{prefix}({name}): property '{prop_name}' missing 'type'"
            # type may be a string or a list (JSON Schema union)
            types_to_check = prop_type if isinstance(prop_type, list) else [prop_type]
            for t in types_to_check:
                if t not in _VALID_JSON_SCHEMA_TYPES:
                    return False, (
                        f"{prefix}({name}): property '{prop_name}' has invalid type {t!r}; "
                        f"must be one of {sorted(_VALID_JSON_SCHEMA_TYPES)}"
                    )

        required = params.get("required")
        if required is not None:
            if not isinstance(required, list):
                return False, f"{prefix}({name}): 'parameters.required' must be a list"
            for req in required:
                if not isinstance(req, str):
                    return False, (
                        f"{prefix}({name}): 'parameters.required' entries must be strings, "
                        f"got {type(req).__name__}"
                    )
                if req not in props:
                    return False, (
                        f"{prefix}({name}): required param '{req}' not in 'properties'"
                    )

    return True, "ok"


# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------

def call_judge_once(
        python_code: str,
        openai_tools: list[dict],
        questions: list[dict],
        task_type: str,
        client: "openai.OpenAI",
        model: str,
        timeout_s: int = 30,
    ) -> dict | None:
    """Call the LLM judge for one tool and return a scores dict or None on error.

    Args:
        python_code:  The generated Python tool code.
        openai_tools: The parsed OpenAI tool schema list.
        questions:    Sample questions (list of {"question": ..., "answer": ...}).
        task_type:    Task type string (e.g. "build").
        client:       An openai.OpenAI instance pointing at the judge endpoint.
        model:        Judge model ID.
        timeout_s:    Per-request timeout in seconds.

    Returns:
        Dict with keys from SCORE_DIMS plus "response_time_s", or None on any error.
    """
    if not JUDGE_SYSTEM_PROMPT:
        logger.warning("judge_utils: JUDGE_SYSTEM_PROMPT is empty; skipping judge call.")
        return None

    # Truncate code if too long
    code_text = python_code or ""
    if len(code_text) > MAX_CODE_CHARS:
        code_text = code_text[:MAX_CODE_CHARS] + "\n# ... [truncated]"

    # Build sample questions block (up to 3)
    sample_lines = []
    for i, q in enumerate(questions[:3], 1):
        sample_lines.append(
            f"Q{i}: {str(q.get('question', ''))[:300]}\n"
            f"    Expected: {q.get('answer', q.get('expected_answer', ''))}"
        )
    sample_block = "\n\n".join(sample_lines) if sample_lines else "(no sample questions available)"

    tools_json = json.dumps(openai_tools, indent=2)

    user_content = (
        f"## Task Context\n"
        f"Task type: {task_type}\n"
        f"\n## Sample problems this tool must solve\n"
        f"{sample_block}\n\n"
        f"## Python Code\n"
        f"```python\n{code_text}\n```\n\n"
        f"## OpenAI Tool Schema\n"
        f"```json\n{tools_json}\n```\n\n"
        f"Now complete all 4 analysis steps, then output your JSON."
    )

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
            max_tokens=1280,
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - t0
        raw = response.choices[0].message.content or ""
        scores = _parse_json_scores(raw)
        scores["response_time_s"] = round(elapsed, 2)
        return scores
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning("judge_utils: judge call failed after %.1fs: %s", elapsed, exc)
        return None


_SUB_DIMS = ["code_correctness", "code_clarity", "schema_quality", "schema_code_alignment"]


def overall_quality_score(scores: dict | None) -> float:
    """Average sub-dimension scores (0-5) normalised to [0, 1]. Returns 0.0 on None/error."""
    if scores is None:
        return 0.0
    try:
        vals = [float(scores[d]) for d in _SUB_DIMS if d in scores]
        return (sum(vals) / len(vals)) / 5.0 if vals else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# __main__ debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import datetime
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Debug judge_utils: fire one judge call and print/save results.")
    parser.add_argument("--base-url", default="http://localhost:9003/v1", help="Judge endpoint base URL")
    parser.add_argument("--model", default="NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4", help="Judge model ID")
    parser.add_argument("--api-key", default="TEST_API")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", default=os.path.join(_HERE, "judge_debug.md"), help="Output markdown file")
    args = parser.parse_args()

    print(f"\n=== judge_utils debug ===")
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"prompt   : {_PROMPT_PATH}")
    print(f"prompt loaded: {bool(JUDGE_SYSTEM_PROMPT)} ({len(JUDGE_SYSTEM_PROMPT)} chars)\n")

    if not JUDGE_SYSTEM_PROMPT:
        print("ERROR: system prompt not loaded — check path above", file=sys.stderr)
        sys.exit(1)

    dummy_code = (
        "def add_numbers(a: float, b: float) -> float:\n"
        "    \"\"\"Return a + b.\"\"\"\n"
        "    return a + b\n"
    )
    dummy_schema = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
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
    }]

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "ERROR"
    parse_error = ""
    scores: dict | None = None
    q = 0.0
    elapsed = 0.0
    raw = ""
    aligned = False
    align_reason = ""

    print("Calling judge...")
    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "## Task Context\nTask type: build\n\n"
                    "## Sample problems this tool must solve\n"
                    "Q1: What is 2 + 3?\n    Expected: 5\n\n"
                    f"## Python Code\n```python\n{dummy_code}```\n\n"
                    f"## OpenAI Tool Schema\n```json\n{json.dumps(dummy_schema, indent=2)}\n```\n\n"
                    "Now complete all 4 analysis steps, then output your JSON."
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
            max_tokens=1280,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - t0
        raw = response.choices[0].message.content or ""
        print(f"\n--- raw response ({elapsed:.1f}s) ---\n{raw}\n---\n")

        try:
            scores = _parse_json_scores(raw)
            q = overall_quality_score(scores)
            aligned, align_reason = check_schema_code_alignment(dummy_code, dummy_schema)
            status = "OK"
            print("parsed scores:", json.dumps(scores, indent=2))
            print(f"\noverall_quality_score → {q:.4f}  (normalised 0-1)")
            print(f"schema-code aligned  : {aligned} ({align_reason})")
        except Exception as parse_exc:
            parse_error = str(parse_exc)
            status = "PARSE_ERROR"
            print(f"PARSE ERROR: {parse_exc}", file=sys.stderr)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        parse_error = str(exc)
        status = "HTTP_ERROR"
        print(f"HTTP/API ERROR after {elapsed:.1f}s: {exc}", file=sys.stderr)

    # --- write to judge_debug.md ---
    header = "| timestamp | model | status | code_correctness | code_clarity | schema_quality | schema_code_alignment | overall_quality_score | aligned | time_s | error |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    def _s(key: str) -> str:
        if scores is None:
            return "—"
        return f"{scores.get(key, '—'):.1f}" if isinstance(scores.get(key), float) else str(scores.get(key, "—"))

    row = (
        f"| {timestamp} | {args.model} | {status} "
        f"| {_s('code_correctness')} | {_s('code_clarity')} | {_s('schema_quality')} | {_s('schema_code_alignment')} "
        f"| {q:.4f} | {aligned} | {elapsed:.1f} | {parse_error or '—'} |\n"
    )

    write_header = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if write_header:
            f.write("# Judge Debug\n\n")
            f.write(header)
        f.write(row)

    print(f"\nResult appended to {args.out}")
    if status != "OK":
        sys.exit(1)
