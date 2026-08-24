"""
Reward functions for tool-making GRPO training.

Tool execution happens inside TRL's multi-turn rollout (via tools.py).
These rewards only inspect the final completion to score correctness,
format quality, and whether the model actually used the tool.
"""
import re
import ast
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tool_rl.llm_judge_eq import llm_as_judge_equal
from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset

logger = logging.getLogger(__name__)

FINAL_ANSWER_TAG_RE = re.compile(r"FINAL[_\s]*ANSWER\s*:", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{")
_warned_missing_env_reward = False
_warned_answer_alignment = False
_warned_question_alignment = False


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def normalize_answer(answer_str):
    """Normalize an answer string for comparison."""
    if answer_str is None:
        return ""
    s = str(answer_str).strip()
    s = s.replace("**", "").replace("*", "")
    s = s.strip("'\"` \n\t")
    s = s.rstrip(".")
    s = " ".join(s.split())
    return s


def answers_match(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    if not pred or not gt:
        return False

    if pred == gt:
        return True

    if pred.lower() == gt.lower():
        return True

    # Integer / hex comparison (handles "0x1a2b" == "6699" etc.)
    try:
        pred_int = int(pred, 0)
        gt_int = int(gt, 0)
        if pred_int == gt_int:
            return True
    except (ValueError, TypeError):
        pass

    # Numeric comparison with relative tolerance so large integers are handled
    # correctly (e.g. 1_000_000 vs 999_999.999_999 matches; 1_000_000 vs
    # 1_000_001 does not).
    try:
        pred_num = float(pred.replace(",", "").replace("$", ""))
        gt_num = float(gt.replace(",", "").replace("$", ""))
        diff = abs(pred_num - gt_num)
        scale = max(abs(pred_num), abs(gt_num), 1.0)
        if diff / scale < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Containment check: only for numeric short answers with word boundaries
    # to avoid false positives like gt="7" matching pred="17 steps".
    if len(gt) <= 10:
        try:
            float(gt.replace(",", "").replace("$", ""))
            # It's a number — require exact match already handled above; skip containment.
        except (ValueError, TypeError):
            # Non-numeric short string: require word-boundary containment.
            import re as _re
            pattern = r"(?<![a-z0-9])" + _re.escape(gt.lower()) + r"(?![a-z0-9])"
            if _re.search(pattern, pred.lower()):
                return True

    return False


def _extract_final_answer(text: str) -> str | None:
    """Pull the answer out of a completion string."""
    if not text:
        return None

    # Strip thinking tags
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # FINAL_ANSWER: ...
    final_line_matches = re.findall(r"FINAL[_\s]*ANSWER\s*:\s*([^\n\r]*)", text, re.IGNORECASE)
    if final_line_matches:
        candidate = final_line_matches[-1].strip()
        if candidate:
            return candidate

    # \boxed{...} — use a brace counter to handle arbitrary nesting depth
    _last_boxed = None
    _search_from = 0
    while True:
        _idx = text.find("\\boxed{", _search_from)
        if _idx == -1:
            break
        _open = _idx + len("\\boxed{") - 1  # position of the opening '{'
        _depth = 0
        _end = -1
        for _i in range(_open, len(text)):
            if text[_i] == "{":
                _depth += 1
            elif text[_i] == "}":
                _depth -= 1
                if _depth == 0:
                    _end = _i
                    break
        if _end != -1:
            _last_boxed = text[_open + 1:_end]
        _search_from = _open + 1
    if _last_boxed is not None:
        return _last_boxed.strip()

    # <answer>...</answer>
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # No fallback parsing: avoid accidentally reading intermediate tool output as final answer.
    return None


# ---------------------------------------------------------------------------
# Cryptarithm verifier
# ---------------------------------------------------------------------------

def _parse_cryptarithm_puzzle(question: str):
    """Extract (words_letters, result_letters, allow_leading_zero) from question text."""
    m = re.search(r'cryptarithm:\s*\n\n(.*?)\n\nEach letter', question, re.DOTALL)
    if not m:
        return None, None, False
    lines = [l for l in m.group(1).strip().split('\n') if l.strip()]
    words, result, past_dashes = [], None, False
    for line in lines:
        s = line.strip()
        if re.match(r'^-+$', s):
            past_dashes = True
            continue
        if past_dashes:
            result = s
            break
        words.append(re.sub(r'^\+\s*', '', s).strip())
    allow_leading_zero = "Leading letters may be zero" in question
    return words, result, allow_leading_zero


def _verify_cryptarithm(mapping: dict, words_letters: list, result_letters: str, allow_leading_zero: bool) -> bool:
    """Return True if mapping satisfies all cryptarithm constraints."""
    all_letters = set()
    for w in words_letters:
        all_letters.update(w)
    all_letters.update(result_letters)
    if set(mapping.keys()) != all_letters:
        return False
    vals = list(mapping.values())
    if any(not isinstance(d, int) or d < 0 or d > 9 for d in vals):
        return False
    if len(set(vals)) != len(vals):
        return False
    if not allow_leading_zero:
        for w in words_letters:
            if w and mapping.get(w[0]) == 0:
                return False
        if result_letters and mapping.get(result_letters[0]) == 0:
            return False
    try:
        nums = [int("".join(str(mapping[c]) for c in w)) for w in words_letters]
        result_num = int("".join(str(mapping[c]) for c in result_letters))
        return sum(nums) == result_num
    except (KeyError, ValueError):
        return False


def _check_cryptarithm_answer(question: str | None, predicted_text: str) -> bool:
    """Return True if predicted_text is a valid solution for the cryptarithm in question."""
    if not question:
        return False
    words, result, allow_lz = _parse_cryptarithm_puzzle(question)
    if not words or not result:
        return False
    mapping = {}
    try:
        for pair in predicted_text.split(','):
            if '=' in pair:
                letter, digit = pair.strip().split('=', 1)
                mapping[letter.strip()] = int(digit.strip())
    except (ValueError, AttributeError):
        return False
    if not mapping:
        return False
    return _verify_cryptarithm(mapping, words, result, allow_lz)


# ---------------------------------------------------------------------------
# Knights & Knaves verifier
# ---------------------------------------------------------------------------

def _check_knights_knaves_answer(predicted_text: str, ground_truth: str) -> bool:
    """Return True if predicted_text is equivalent to ground_truth for K&K puzzles."""
    try:
        oracle = KnightsKnavesDataset._normalize_answer(ground_truth)
        predicted = KnightsKnavesDataset._normalize_answer(predicted_text)
        return bool(oracle) and oracle == predicted
    except Exception:
        return False


def _get_text(completion):
    """Unwrap a completion that may be conversational or plain text."""
    if isinstance(completion, list):
        # Conversational: take last assistant message
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        completion_str = completion[-1].get("content", "") if completion else ""
        if '</tool_response>' in completion_str:
            return completion_str.split('</tool_response>')[-1]
        return completion_str
    completion_str = str(completion)
    if '</tool_response>' in completion_str:
        return completion_str.split('</tool_response>')[-1]
    return completion_str


def _align_ground_truth_answers(completions, answer):
    """Align answer list to completion count to avoid silent zip truncation/misalignment."""
    global _warned_answer_alignment
    n = len(completions)

    if answer is None:
        return ["" for _ in range(n)]

    if isinstance(answer, (list, tuple)):
        answers = list(answer)
    else:
        answers = [answer]

    m = len(answers)
    if m == n:
        return answers
    if m == 0:
        return ["" for _ in range(n)]

    # Most common mismatch under repeated generations: one GT repeated per group.
    if n % m == 0:
        repeat_factor = n // m
        if not _warned_answer_alignment:
            logger.warning(
                "Reward alignment: completions=%d, answers=%d; expanding each answer by factor %d.",
                n,
                m,
                repeat_factor,
            )
            _warned_answer_alignment = True
        return [a for a in answers for _ in range(repeat_factor)]

    if not _warned_answer_alignment:
        logger.warning(
            "Reward alignment: completions=%d, answers=%d; using truncate/pad fallback.",
            n,
            m,
        )
        _warned_answer_alignment = True
    if m > n:
        return answers[:n]
    return answers + [answers[-1]] * (n - m)


def _align_questions(completions, question):
    """Align question list to completion count to avoid mismatch diagnostics noise."""
    global _warned_question_alignment
    n = len(completions)

    if question is None:
        return [None for _ in range(n)]
    if isinstance(question, (list, tuple)):
        questions = list(question)
    else:
        questions = [question]

    m = len(questions)
    if m == n:
        return questions
    if m == 0:
        return [None for _ in range(n)]

    if n % m == 0:
        repeat_factor = n // m
        if not _warned_question_alignment:
            logger.warning(
                "Question alignment: completions=%d, questions=%d; expanding each question by factor %d.",
                n,
                m,
                repeat_factor,
            )
            _warned_question_alignment = True
        return [q for q in questions for _ in range(repeat_factor)]

    if not _warned_question_alignment:
        logger.warning(
            "Question alignment: completions=%d, questions=%d; using truncate/pad fallback.",
            n,
            m,
        )
        _warned_question_alignment = True
    if m > n:
        return questions[:n]
    return questions + [questions[-1]] * (n - m)


_QUESTION_ARG_KEYS = ("question", "problem_text", "problem", "query", "prompt", "input", "text")


def _normalize_question_text(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _extract_question_like_args_from_payload(payload: dict) -> list[str]:
    vals: list[str] = []

    def add_val(v):
        if isinstance(v, str):
            v = v.strip()
            if v:
                vals.append(v)

    if not isinstance(payload, dict):
        return vals

    args = payload.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    if isinstance(args, dict):
        for k, v in args.items():
            k = str(k).lower()
            if any(tok in k for tok in _QUESTION_ARG_KEYS):
                add_val(v)

    # Support payloads where args may already be flattened.
    for key in _QUESTION_ARG_KEYS:
        if key in payload:
            add_val(payload.get(key))

    return vals


def _context_shifted_from_expected_question(completion, expected_question: str | None) -> bool:
    """Return True if tool-call question args exist and do not match expected question."""
    expected_norm = _normalize_question_text(expected_question)
    if not expected_norm:
        return False

    candidates: list[str] = []
    for payload in _iter_tool_payloads(completion):
        if payload.get("_invalid_json"):
            continue
        candidates.extend(_extract_question_like_args_from_payload(payload))

    # If no question-like args were found, don't force a mismatch.
    if not candidates:
        return False

    for cand in candidates:
        cand_norm = _normalize_question_text(cand)
        if not cand_norm:
            continue
        if cand_norm == expected_norm:
            return False
        # Allow wrapper text around the same question.
        if len(cand_norm) >= 20 and (cand_norm in expected_norm or expected_norm in cand_norm):
            return False
    return True


def _iter_tool_payloads(completion):
    """Yield tool payload dicts from either structured tool_calls or <tool_call> blocks."""
    # Structured tool_calls (two-tool workflow)
    if isinstance(completion, list):
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict) or tc.get("type") != "function":
                    continue
                func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        yield {"_invalid_json": True, "_raw_arguments": args, "_tool_name": func.get("name")}
                        continue
                if isinstance(args, dict):
                    payload = dict(args)
                    payload["_tool_name"] = func.get("name")
                    yield payload
        # Also parse any <tool_call> blocks in the last assistant message
        text = _get_text(completion)
    else:
        text = _get_text(completion)

    if not text:
        return

    # Custom <tool_call> JSON blocks
    for raw in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload
        except json.JSONDecodeError:
            yield {"_invalid_json": True, "_raw_arguments": raw}


# ---------------------------------------------------------------------------
# GRPO reward functions
# ---------------------------------------------------------------------------

def correctness_reward(prompts, completions, answer, **kwargs):
    """Check whether the model's final answer is correct.

    This is the primary training signal.  Full marks (2.0) if the final
    answer extracted from the completion matches the ground truth, partial
    credit (0.5) if the ground truth string appears anywhere in the last
    assistant turn.
    """
    n = len(completions)
    rewards = [None] * n
    aligned_answers = _align_ground_truth_answers(completions, answer)
    aligned_questions = _align_questions(completions, kwargs.get("question"))

    # Pass 1: cheap checks; collect indices that need LLM judge.
    # Each pending entry: (index, expr1, expr2, reward_if_yes)
    judge_pending: list[tuple[int, str, str, float]] = []

    for i, (completion, gt, expected_question) in enumerate(
        zip(completions, aligned_answers, aligned_questions, strict=False)
    ):
        if _context_shifted_from_expected_question(completion, expected_question):
            rewards[i] = 0.0
            continue

        text = _get_text(completion)
        if '</tool_response>' in text:
            text = text.split('</tool_response>')[-1]
        gt = str(gt) if gt is not None else ""
        gt_norm = normalize_answer(gt)
        if not gt_norm:
            rewards[i] = 0.0
            continue
        final = _extract_final_answer(text)
        if final:
            if answers_match(final, gt):
                rewards[i] = 2.0
            elif _check_cryptarithm_answer(expected_question, final):
                rewards[i] = 2.0
            elif _check_knights_knaves_answer(final, gt):
                rewards[i] = 2.0
            else:
                judge_pending.append((i, final, gt, 1.0, expected_question))
        else:
            # No final answer tag found — don't send the full completion text to
            # the expression-equivalence judge (it expects two short expressions).
            # Treat missing final answer as incorrect.
            rewards[i] = 0.0

    # Pass 2: run all LLM judge calls in parallel.
    if judge_pending:
        max_workers = min(len(judge_pending), 64)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(llm_as_judge_equal, expr1, expr2): (idx, reward_yes, question, expr1, expr2)
                for idx, expr1, expr2, reward_yes, question in judge_pending
            }
            for fut in as_completed(futures):
                idx, reward_yes, question, expr1, expr2 = futures[fut]
                judge_result = fut.result()
                logger.debug("Judge mismatch question=%s", question)
                logger.debug("Judge compare expr1=%s expr2=%s result=%s", expr1[:100], expr2, judge_result)

                try:
                    rewards[idx] = reward_yes if judge_result else 0.0
                except Exception:
                    rewards[idx] = 0.0

    return rewards


def _find_tool_names_in_completion(completion):
    """Extract set of tool names used in a conversational completion."""
    names = set()
    if not isinstance(completion, list):
        return names
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        # Assistant messages with tool_calls
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict) and tc.get("type") == "function":
                names.add(tc["function"]["name"])
        # Tool-result messages
        if msg.get("role") == "tool" and msg.get("name"):
            names.add(msg["name"])
    return names


def tool_usage_reward(prompts, completions, **kwargs):
    """Reward the model for following the two-phase tool workflow.

    Full marks (1.0) if the model called both create_tool AND call_tool.
    Partial credit (0.3) if it called only create_tool (wrote a tool but
    didn't execute it).
    """
    rewards = []
    for completion in completions:
        names = _find_tool_names_in_completion(completion)
        created = "create_tool" in names
        called = "call_tool" in names
        if created and called:
            rewards.append(1.0)
        elif created:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """Light bonus for providing a FINAL_ANSWER line or \\boxed{} / \\box{} answer."""
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        has_answer = bool(FINAL_ANSWER_TAG_RE.search(text)) or bool(_BOXED_RE.search(text))
        rewards.append(0.5 if has_answer else 0.0)
    return rewards


def env_reward_reward(prompts, completions, env_reward=None, **kwargs):
    """Pass through env_reward from rollout_func if provided."""
    global _warned_missing_env_reward
    if env_reward is None:
        if not _warned_missing_env_reward:
            logger.warning("env_reward not provided to reward function; returning 0.0 for all samples.")
            _warned_missing_env_reward = True
        return [0.0 for _ in completions]

    if isinstance(env_reward, (list, tuple)):
        rewards = list(env_reward)
    else:
        try:
            val = float(env_reward)
        except (TypeError, ValueError):
            val = 0.0
        rewards = [val for _ in completions]

    if len(rewards) != len(completions):
        if not _warned_missing_env_reward:
            logger.warning(
                "env_reward length %d != completions length %d; truncating/padding with 0.0.",
                len(rewards),
                len(completions),
            )
            _warned_missing_env_reward = True
        if len(rewards) < len(completions):
            rewards = rewards + [0.0] * (len(completions) - len(rewards))
        else:
            rewards = rewards[: len(completions)]

    out = []
    for r in rewards:
        if r is None:
            out.append(0.0)
            continue
        try:
            out.append(float(r))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def build_env_reward(prompts, completions, env_reward=None, **kwargs):
    """Pass-through env_reward for build tasks (avg test accuracy)."""
    return env_reward_reward(prompts, completions, env_reward=env_reward, **kwargs)


def use_exec_reward(prompts, completions, env_reward=None, **kwargs):
    """Pass-through env_reward for use tasks (tool execution signal)."""
    return env_reward_reward(prompts, completions, env_reward=env_reward, **kwargs)


def use_correct_reward(prompts, completions, use_correct_reward=None, **kwargs):
    """Pass-through efficiency-weighted correctness reward for use tasks."""
    # Avoid duplicate keyword collisions when rollout returns both env_reward and use_correct_reward.
    kwargs.pop("env_reward", None)
    return env_reward_reward(prompts, completions, env_reward=use_correct_reward, **kwargs)


def judge_build_reward(prompts, completions, judge_reward=None, **kwargs):
    """Pass-through LLM judge quality score for build tasks.

    Receives the ``judge_reward`` list emitted by
    ``decompose_rollout_w_judge``.  Returns 0.0 for all completions when
    ``judge_reward`` is absent (i.e. when using a rollout that does not emit
    judge scores), so this function is safe to register for any rollout.
    """
    # Drop env_reward from kwargs to avoid a duplicate keyword argument error:
    # the rollout return dict contains env_reward which TRL passes through as
    # a kwarg, and env_reward_reward also accepts env_reward as an explicit arg.
    kwargs.pop("env_reward", None)
    return env_reward_reward(prompts, completions, env_reward=judge_reward, **kwargs)


def _tool_penalty_components(
    completion,
    code_penalty=1.0,
    json_penalty=0.5,
    schema_penalty=0.5,
):
    """Return separate (python_syntax_penalty, json_schema_penalty) components."""
    python_penalty = 0.0
    json_schema_penalty = 0.0

    for payload in _iter_tool_payloads(completion):
        if payload.get("_invalid_json"):
            json_schema_penalty += json_penalty
            continue

        code = payload.get("code")
        if isinstance(code, str):
            try:
                ast.parse(code)
            except SyntaxError:
                python_penalty += code_penalty

        if "arguments" in payload:
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    logger.debug("Invalid tool arguments JSON: %s", arguments)
                    json_schema_penalty += json_penalty
            elif arguments is not None and not isinstance(arguments, dict):
                json_schema_penalty += json_penalty

        for key in ("schema", "json_schema", "tool_schema"):
            if key in payload:
                val = payload.get(key)
                if isinstance(val, str):
                    try:
                        json.loads(val)
                    except json.JSONDecodeError:
                        logger.debug("Invalid schema field %s: %s", key, val)
                        json_schema_penalty += schema_penalty
                elif val is not None and not isinstance(val, dict):
                    json_schema_penalty += schema_penalty

    # Also validate fenced python/json blocks (tool creation step)
    text = _get_text(completion)
    if text:
        for py_code in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL):
            try:
                ast.parse(py_code)
            except SyntaxError:
                python_penalty += code_penalty
        for json_code in re.findall(r"```json\n(.*?)\n```", text, re.DOTALL):
            try:
                json.loads(json_code)
            except json.JSONDecodeError:
                logger.debug("Block JSON penalty applied")
                json_schema_penalty += schema_penalty

    return python_penalty, json_schema_penalty


def tool_python_syntax_penalty(
    prompts,
    completions,
    code_penalty=1.0,
    max_penalty=2.0,
    **kwargs,
):
    """Penalty for invalid Python syntax in tool code payloads."""
    penalties = []
    for completion in completions:
        python_penalty, _ = _tool_penalty_components(
            completion,
            code_penalty=code_penalty,
            json_penalty=0.0,
            schema_penalty=0.0,
        )
        penalties.append(-min(max_penalty, python_penalty))
    return penalties


def tool_json_schema_penalty(
    prompts,
    completions,
    json_penalty=0.5,
    schema_penalty=0.5,
    max_penalty=2.0,
    **kwargs,
):
    """Penalty for invalid JSON arguments and malformed JSON schema payloads."""
    penalties = []
    for completion in completions:
        _, json_schema = _tool_penalty_components(
            completion,
            code_penalty=0.0,
            json_penalty=json_penalty,
            schema_penalty=schema_penalty,
        )
        penalties.append(-min(max_penalty, json_schema))
    return penalties


def tool_syntax_schema_penalty(
    prompts,
    completions,
    code_penalty=1.0,
    json_penalty=0.5,
    schema_penalty=0.5,
    max_penalty=2.0,
    **kwargs,
):
    """Backward-compatible combined penalty (python syntax + json/schema)."""
    penalties = []
    for completion in completions:
        python_penalty, json_schema = _tool_penalty_components(
            completion,
            code_penalty=code_penalty,
            json_penalty=json_penalty,
            schema_penalty=schema_penalty,
        )
        penalties.append(-min(max_penalty, python_penalty + json_schema))
    return penalties


def final_answer_spam_penalty(prompts, completions, max_penalty=2.0, per_extra=0.5, **kwargs):
    """Penalty for repeated FINAL_ANSWER tags or spammy final answers."""
    penalties = []
    for completion in completions:
        text = _get_text(completion)
        if not text:
            penalties.append(0.0)
            continue
        fa_count = len(FINAL_ANSWER_TAG_RE.findall(text))
        boxed_count = len(_BOXED_RE.findall(text))
        count = max(fa_count, boxed_count)
        logger.debug("Final answer spam count=%s fa_tags=%s boxed=%s", count, fa_count, boxed_count)
        if count <= 1:
            penalties.append(0.0)
            continue
        penalty = -min(max_penalty, per_extra * (count - 1))
        penalties.append(penalty)
    return penalties


# ---------------------------------------------------------------------------
# PPR-style process reward with ReNorm (optional, opt-in only)
# ---------------------------------------------------------------------------

def _score_step1_python(completion) -> float:
    """Score create_tool step: Python code quality in [0, 1].

    Returns 0.5 for syntactically valid code, 1.0 if it also executes
    without error (runtime check), 0.0 if syntax is invalid or no code found.
    """
    code_blocks: list[str] = []

    if isinstance(completion, list):
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict) or tc.get("type") != "function":
                    continue
                func = tc.get("function", {})
                if func.get("name") != "create_tool":
                    continue
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return 0.0
                code = args.get("code", "") if isinstance(args, dict) else ""
                if code:
                    code_blocks.append(code)

    text = _get_text(completion)
    if text:
        for block in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL):
            code_blocks.append(block)

    if not code_blocks:
        return 0.0

    code = code_blocks[-1]
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    try:
        import io as _io
        import contextlib as _ctx
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf), _ctx.redirect_stderr(buf):
            exec(compile(code, "<tool>", "exec"), {})
        return 1.0
    except Exception:
        return 0.5  # valid syntax but runtime error


def _score_step2_json(completion) -> float:
    """Score call_tool step: JSON arguments + schema validity in [0, 1].

    Returns 0.5 if arguments are valid JSON, +0.5 if a schema field is also
    valid JSON/dict. Returns 0.0 if no call_tool invocation found or JSON
    is malformed.
    """
    call_payloads: list[dict] = []

    if isinstance(completion, list):
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict) or tc.get("type") != "function":
                    continue
                func = tc.get("function", {})
                if func.get("name") != "call_tool":
                    continue
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return 0.0
                if isinstance(args, dict):
                    call_payloads.append(args)

    text = _get_text(completion)
    if not call_payloads and text:
        for block in re.findall(r"```json\n(.*?)\n```", text, re.DOTALL):
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    call_payloads.append(obj)
            except json.JSONDecodeError:
                return 0.0

    if not call_payloads:
        return 0.0

    payload = call_payloads[-1]
    args_score = 0.5  # already parsed → valid JSON

    schema_score = 0.0
    for key in ("schema", "json_schema", "tool_schema"):
        val = payload.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            schema_score = 0.5
            break
        if isinstance(val, str):
            try:
                json.loads(val)
                schema_score = 0.5
            except json.JSONDecodeError:
                pass
            break

    return args_score + schema_score


def process_renorm_reward(
    prompts,
    completions,
    answer=None,
    step1_weight: float = 0.5,
    step2_weight: float = 0.5,
    **kwargs,
):
    """Optional PPR-style process reward with ReNorm.

    NOT used by default. Opt in by adding 'process_renorm_reward' to the
    reward_funcs list in your training config alongside (or instead of) the
    individual penalty functions.

    Applies the ReNorm formula from arXiv:2509.25598 (PPR paper):

        r_p = r̂_p + r_o - 1   ∈ [-1, 1]

    where r̂_p is a weighted average of two verifiable step scores:
      - Step 1 (create_tool): Python code syntax + execution quality ∈ [0, 1]
      - Step 2 (call_tool):   JSON arguments + schema validity ∈ [0, 1]
    and r_o ∈ {0, 1} is the binarized outcome reward.

    Effect:
      - Correct trajectory + good steps  → r_p ∈ (0,  1]  (positive)
      - Correct trajectory + bad steps   → r_p ∈ [-1, 0)  (still penalized)
      - Wrong  trajectory + good steps   → r_p ∈ [-1, 0]  (no credit)
      - Wrong  trajectory + bad steps    → r_p = -1        (max penalty)
    """
    outcome_rewards = correctness_reward(prompts, completions, answer, **kwargs)

    rewards = []
    for completion, r_o_raw in zip(completions, outcome_rewards):
        r_o = 1.0 if (r_o_raw or 0.0) > 0 else 0.0  # binarize

        r_p1 = _score_step1_python(completion)
        r_p2 = _score_step2_json(completion)

        r_hat_p = step1_weight * r_p1 + step2_weight * r_p2  # ∈ [0, 1]
        r_p = r_hat_p + r_o - 1.0                            # ∈ [-1, 1]

        logger.debug(
            "process_renorm: r_p1=%.2f r_p2=%.2f r_hat_p=%.2f r_o=%.1f r_p=%.2f",
            r_p1, r_p2, r_hat_p, r_o, r_p,
        )
        rewards.append(r_p)

    return rewards
