"""
Two-phase tool executor for GRPO tool-making training.

Mirrors the tool_robustness pipeline (e.g. ab.py) where:
  Phase 1 – create_tool:  Model writes Python code → validated & stored
  Phase 2 – call_tool:    Model calls the stored tool with arguments → sandbox executes

TRL handles the multi-turn loop automatically:
  1. Model generates a tool_call to `create_tool` with its Python code
  2. TRL calls create_tool → code is validated & registered under a session_id
  3. TRL injects the result (including session_id) into the conversation
  4. Model generates a tool_call to `call_tool` with the session_id + arguments
  5. TRL calls call_tool → sandbox executes → result returned
  6. Model reads result and provides final answer (or iterates)

Token-level logprobs for tool-result tokens are masked via TRL's `tool_mask`.
"""
import os
import io
import json
import sys
import uuid
import traceback
import contextlib
import subprocess
import multiprocessing
import requests
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sandbox execution backends
# ---------------------------------------------------------------------------

SANDBOX_URL = os.environ.get("TOOL_RL_SANDBOX_URL", "http://localhost:9080/run_code")
USE_LOCAL_EXEC = os.environ.get("TOOL_RL_USE_SANDBOX", "0") != "1"


def _exec_code_worker(code: str, result_queue: multiprocessing.Queue):
    """Run code in a child process (isolation + timeout)."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, {"__builtins__": __builtins__}, {})
        result_queue.put({
            "status": "Success",
            "run_result": {"stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()},
        })
    except Exception:
        result_queue.put({
            "status": "Error",
            "run_result": {"stdout": stdout_buf.getvalue(), "stderr": traceback.format_exc()},
        })


def _run_code_local(code: str, timeout: int = 10) -> dict:
    """Execute code in a subprocess with timeout.

    Uses subprocess.run() instead of multiprocessing.Process so that the child
    immediately exec()s a fresh Python interpreter.  This avoids the fork-without-exec
    hazard present when the parent has many threads: any mutex held by another thread
    at fork time is permanently locked in the child, causing deadlocks inside
    multiprocessing.Queue and leaving futures unresolved.  subprocess.run() is also
    fork-safe because exec() clears inherited mutexes, and its timeout is reliable.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        status = "Success" if result.returncode == 0 else "Error"
        return {"status": status, "run_result": {"stdout": result.stdout, "stderr": result.stderr}}
    except subprocess.TimeoutExpired:
        return {"status": "Error", "run_result": {"stdout": "", "stderr": "Execution timed out"}}
    except Exception as exc:
        return {"status": "Error", "run_result": {"stdout": "", "stderr": str(exc)}}


def _run_code_sandbox(code: str, timeout: int = 20) -> dict:
    """Execute code via the Docker sandbox server."""
    try:
        resp = requests.post(
            SANDBOX_URL,
            headers={"Content-Type": "application/json"},
            json={"code": code, "language": "python"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return _run_code_local(code, timeout=timeout)


def run_code(code: str, timeout: int = 20) -> dict:
    if USE_LOCAL_EXEC:
        return _run_code_local(code, timeout=timeout)
    return _run_code_sandbox(code, timeout=timeout)


# ---------------------------------------------------------------------------
# Session-based tool registry (UUID-keyed to isolate parallel completions)
# ---------------------------------------------------------------------------

_tool_registry: dict[str, dict] = {}


def _cleanup_registry(max_size: int = 10000):
    """Prevent unbounded growth during long training runs."""
    if len(_tool_registry) > max_size:
        # Remove oldest half
        keys = list(_tool_registry.keys())
        for k in keys[: len(keys) // 2]:
            del _tool_registry[k]


# ---------------------------------------------------------------------------
# Phase 1: create_tool – model writes a Python tool
# ---------------------------------------------------------------------------

def create_tool(code: str, function_name: str) -> str:
    """Register a Python function as a reusable tool.

    Write a complete Python function definition in the ``code`` parameter and
    specify which function to expose via ``function_name``.  The code is
    validated (syntax check) and stored.  A unique ``session_id`` is returned
    that you **must** pass to ``call_tool`` to execute the function.

    Args:
        code: Complete Python source code containing one or more function
            definitions.  Must be valid, self-contained Python using only the
            standard library.  Example:
            ``def solve(sequence):\\n    ...\\n    return next_char``
        function_name: The name of the function defined in ``code`` to expose
            as the tool entry point.

    Returns:
        A confirmation message containing the ``session_id`` to use with
        ``call_tool``, or an error message if validation failed.
    """
    # Syntax validation
    try:
        compile(code, "<tool>", "exec")
    except SyntaxError as e:
        return f"Syntax error in tool code: {e}"

    # Check that the declared function exists in the code
    if f"def {function_name}" not in code:
        return f"Error: function '{function_name}' not found in code."

    # Register under a unique session id
    session_id = uuid.uuid4().hex[:8]
    _tool_registry[session_id] = {
        "code": code,
        "function_name": function_name,
    }
    _cleanup_registry()

    return (
        f"Tool '{function_name}' registered successfully. "
        f"session_id={session_id}  "
        f"Use call_tool(session_id=\"{session_id}\", arguments=\"...\") to execute it."
    )


# ---------------------------------------------------------------------------
# Phase 2: call_tool – model executes the tool it created
# ---------------------------------------------------------------------------

def call_tool(session_id: str, arguments: str) -> str:
    """Execute a previously created tool with the given arguments.

    Runs the function that was registered via ``create_tool`` in a sandboxed
    environment.  You must provide the ``session_id`` returned by
    ``create_tool`` and a JSON string of keyword arguments.

    Args:
        session_id: The session identifier returned by ``create_tool``.
        arguments: A JSON-encoded string of keyword arguments to pass to
            the function.  Example: ``{"sequence": "a b a b a b"}``

    Returns:
        The string representation of the function's return value, or an
        error message if execution failed.
    """
    entry = _tool_registry.get(session_id)
    if entry is None:
        available = list(_tool_registry.keys())[-5:]  # show last few
        return (
            f"Error: session_id '{session_id}' not found. "
            f"Available sessions: {available}. "
            f"Create a tool first with create_tool."
        )

    code = entry["code"]
    function_name = entry["function_name"]

    # Parse arguments
    try:
        args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError) as exc:
        return f"Error parsing arguments JSON: {exc}"

    if not isinstance(args_dict, dict):
        return f"Error: arguments must be a JSON object, got {type(args_dict).__name__}"

    # Build execution script
    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args_dict.items())
    script = f"""{code}

_result = {function_name}({args_str})
print(_result)
"""
    result = run_code(script)

    if result.get("status") == "Success":
        stdout = result["run_result"]["stdout"].strip()
        return stdout if stdout else "(no output)"
    else:
        stderr = result.get("run_result", {}).get("stderr", "Unknown error")
        if len(stderr) > 500:
            stderr = stderr[-500:]
        return f"Execution error: {stderr}"
