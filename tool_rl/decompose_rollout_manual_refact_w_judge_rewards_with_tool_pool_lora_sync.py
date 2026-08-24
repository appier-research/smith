"""
Decomposed rollout with two reward signals for tool-building and colocated rollout for tool-use.

Task types
----------
- use:   Model uses an existing tool to solve a question. Rollout runs on the colocated
         vLLM engine (the policy being trained). Reward is rule-based answer correctness.
- build: Model writes tool code + schema. Two independent reward signals:

    1. build_eval_reward (tool usability correctness):
       decompose_single_q_rollout_external runs the built tool against test questions
       using a lag-behind vLLM serving model at --evaluator_base_url (default
       localhost:9002). This model is periodically synced with the latest LoRA
       checkpoint via _LoraJudgeSyncCallback, providing a stable but improving target.
       trainer.evaluator_model is updated after each sync.

    2. judge_reward (tool semantic readability / code quality):
       call_judge_once queries a fixed high-quality judge (--judge_model, e.g.
       Qwen3-30B at --judge_base_url). This model is NEVER updated during training
       and scores code_correctness, code_clarity, schema_quality, and alignment.

Trainer attributes consumed (all optional; graceful degradation when absent):
  trainer.use_judge_reward      bool   Master switch (default False → no-op)
  trainer.judge_base_url        str    Fixed judge endpoint URL (code quality)
  trainer.judge_model           str    Fixed judge model ID — never changes
  trainer.judge_api_key         str    API key for the judge endpoint
  trainer.judge_reward_weight   float  Weight for judge_build_reward axis
  trainer.judge_max_workers     int    Thread-pool size for parallel judge calls
  trainer.judge_timeout_s       int    Per-call timeout in seconds
  trainer.judge_max_questions   int    Max questions evaluated per build state
  trainer.evaluator_base_url    str    Lag-behind evaluator endpoint URL (usability)
  trainer.evaluator_model       str    Current evaluator model/adapter name; updated
                                       by _LoraJudgeSyncCallback after each checkpoint
  trainer.evaluator_api_key     str    API key for the evaluator endpoint
  trainer.judge_lora_version    int    Incremented by _LoraJudgeSyncCallback each sync

Metadata is embedded in the prompt with <tool_rl_metadata>{...}</tool_rl_metadata>
and stripped before sending to vLLM so the model never sees hidden answers.
"""

import json
import logging
import os
import random
import re
import threading
import uuid
import math as _math
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeoutError
from typing import Any

import openai

from tool_rl.rewards import answers_match
from tool_rl.judge_utils import call_judge_once, overall_quality_score, check_schema_code_alignment, validate_openai_tool_schema
from tool_rl.xgrammar_schema_check import check_schema
from tool_rl.decompose_single_q_rollout_external import decompose_single_q_rollout_external
from tool_rl.decompose_rollout_utils import (
    TOOL_EXEC_SUCCESS_REWARD,
    _strip_metadata, extract_code_blocks, _apply_chat_template,
    _parse_chatml_messages, _normalize_tool_schema, _extract_tool_names,
    _extract_sampled_logprobs, _remove_dunder_main_block,
    _extract_function_names_from_code, _extract_function_params_from_code,
    _extract_function_params_map, _extract_param_names_from_schema,
    _remap_call_args, _coerce_call_args, _filter_valid_tool_calls,
    extract_tool_calls_from_text, _execute_tool_job,
    _safe_preview, _format_messages_for_log, _maybe_print_messages,
    _get_rollout_logger, _get_llm_engine,
    _has_exactly_one_python_and_one_json_block,
    _FINAL_ANSWER_PROMPT_PREFIX, _format_user_question, _extract_final_answer,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _score_build_task(
        *,
        openai_tools: list[dict],
        python_code: str,
        tool_timeout: int,
        test_set: list[dict],
        tool_eval_parallel: bool,
        parallel_tool_workers: int,
        tool_metrics: dict[str, Any] | None = None,
        # --- External OpenAI-compatible evaluator mode ---
        # Uses decompose_single_q_rollout_external with a lag-behind vLLM serving
        # model (periodically LoRA-synced). Active when evaluator_client is provided.
        evaluator_client=None,
        evaluator_model: str | None = None,
        max_rounds: int = 5,
        # --- Colocated vLLM engine mode ---
        # Uses llm_engine.generate() directly with the training model. Active when
        # llm_engine is provided. Both modes are first-class (ablation study).
        llm_engine=None,
        processing_class=None,
        tool_name: str | None = None,
        function_name: str | None = None,
        max_iters: int = 5,
        tool_sampling_params=None,
        partial_final_answer_weight: float = 0.0,
    ) -> float:
    """Score a build task by running test questions through either the external evaluator
    or the colocated vLLM engine. Both modes are valid; the caller selects by providing
    the appropriate params (ablation study).
    """
    if not test_set:
        return 0.0

    # ---------------------------------------------------------------------------
    # External OpenAI-compatible evaluator mode
    # ---------------------------------------------------------------------------
    if evaluator_client is not None or evaluator_model is not None:
        def _eval_one(t_row: dict) -> tuple[bool, dict]:
            try:
                result = decompose_single_q_rollout_external(
                    question=t_row["question"],
                    python_code=python_code,
                    json_schema=openai_tools,
                    max_rounds=max_rounds,
                    evaluator_client=evaluator_client,
                    evaluator_model=evaluator_model,
                    tool_timeout=tool_timeout,
                )
            except Exception as exc:
                logger.debug("_score_build_task evaluator call failed: %s", exc)
                return False, {}
            stats = result.get("stats") or {}
            tool_used = bool(stats.get("tool_calls_successful", 0))
            predicted = result.get("extracted_answer")
            correct = (
                tool_used
                and predicted is not None
                and answers_match(str(predicted), str(t_row["answer"]))
            )
            return correct, result

        if tool_eval_parallel and len(test_set) > 1:
            max_workers = min(parallel_tool_workers, len(test_set))
            # Per-question upper bound: each question runs max_rounds of (HTTP call +
            # subprocess), each capped at tool_timeout seconds.  With max_workers
            # running in parallel, the total expected time is:
            #   ceil(n_questions / max_workers) * max_rounds * 2 * tool_timeout
            _n_batches = _math.ceil(len(test_set) / max_workers)
            _inner_timeout = max(_n_batches * max_rounds * 2 * tool_timeout + 30, 60)
            _inner_executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                _inner_futs = {_inner_executor.submit(_eval_one, t_row): i
                               for i, t_row in enumerate(test_set)}
                _eval_results_map: dict[int, tuple] = {}
                try:
                    for _fut in as_completed(_inner_futs, timeout=_inner_timeout):
                        _idx = _inner_futs[_fut]
                        try:
                            _eval_results_map[_idx] = _fut.result()
                        except Exception:
                            _eval_results_map[_idx] = (False, {})
                except _FuturesTimeoutError:
                    logger.warning(
                        "_score_build_task: inner eval timed out after %ds "
                        "(%d/%d questions answered); treating missing as incorrect.",
                        _inner_timeout, len(_eval_results_map), len(test_set),
                    )
                    for _fut in _inner_futs:
                        _fut.cancel()
            finally:
                # wait=False: don't block on threads stuck in HTTP calls; they'll
                # finish on their own once their per-request timeout fires (≤tool_timeout s).
                _inner_executor.shutdown(wait=False, cancel_futures=True)
            eval_results = [_eval_results_map.get(i, (False, {})) for i in range(len(test_set))]
        else:
            eval_results = [_eval_one(t_row) for t_row in test_set]

        total_score = 0
        tool_call_count = 0
        tool_success_call_count = 0
        tool_error_call_count = 0
        tool_called_items = 0
        tool_success_items = 0

        for correct, result in eval_results:
            if correct:
                total_score += 1
                tool_success_items += 1
            stats = result.get("stats") or {}
            made = stats.get("tool_calls_made", 0)
            tool_call_count += made
            tool_success_call_count += stats.get("tool_calls_successful", 0)
            tool_error_call_count += stats.get("tool_calls_failed", 0)
            if made > 0:
                tool_called_items += 1

        if tool_metrics is not None:
            tool_metrics["tool_call_count"] = tool_call_count
            tool_metrics["tool_success_call_count"] = tool_success_call_count
            tool_metrics["tool_error_call_count"] = tool_error_call_count
            tool_metrics["tool_called_items"] = tool_called_items
            tool_metrics["tool_success_items"] = tool_success_items
            tool_metrics["too_long_prompt_count"] = 0
            tool_metrics["total_items"] = len(test_set)

        return total_score

    # ---------------------------------------------------------------------------
    # Colocated vLLM engine mode
    # ---------------------------------------------------------------------------
    expected_param_names = (
        _extract_param_names_from_schema(openai_tools)
        or _extract_function_params_from_code(python_code)
    )
    items: list[dict[str, Any]] = []
    for t_row in test_set:
        question, answer = t_row["question"], t_row["answer"]
        messages = [{"role": "user", "content": _format_user_question(question)}]
        prompt = _apply_chat_template(
            processing_class,
            messages,
            tools=openai_tools,
            add_generation_prompt=True,
        )
        items.append({"answer": answer, "messages": messages, "prompt": prompt,
                      "done": False, "tool_called": False, "tool_success": False})
    while len(items) < 5:
        for t_row in test_set:
            question, answer = t_row["question"], t_row["answer"]
            messages = [{"role": "user", "content": _format_user_question(question)}]
            prompt = _apply_chat_template(
                processing_class,
                messages,
                tools=openai_tools,
                add_generation_prompt=True,
            )
            items.append({"answer": answer, "messages": messages, "prompt": prompt,
                          "done": False, "tool_called": False, "tool_success": False})

    try:
        max_model_len = llm_engine.llm_engine.model_config.max_model_len
    except AttributeError:
        max_model_len = None

    total_score = 0
    tool_call_count = 0
    tool_success_call_count = 0
    tool_error_call_count = 0
    too_long_count = 0
    for _ in range(max_iters):
        active_idxs = [idx for idx, item in enumerate(items) if not item["done"]]
        if not active_idxs:
            break

        if max_model_len is not None:
            filtered_idxs = []
            for idx in active_idxs:
                token_len = len(processing_class.encode(items[idx]["prompt"]))
                if token_len > (max_model_len * 0.8):
                    items[idx]["done"] = True
                    too_long_count += 1
                else:
                    filtered_idxs.append(idx)
            active_idxs = filtered_idxs
            if not active_idxs:
                break

        outputs = llm_engine.generate(
            prompts=[items[idx]["prompt"] for idx in active_idxs],
            sampling_params=tool_sampling_params,
        )

        tool_jobs: list[dict[str, Any]] = []
        for item_idx, output in zip(active_idxs, outputs, strict=False):
            item = items[item_idx]
            step_text = processing_class.decode(output.outputs[0].token_ids, skip_special_tokens=False)

            final_answer = _extract_final_answer(step_text)
            if final_answer is not None:
                if answers_match(final_answer, item["answer"]):
                    if item.get("tool_success"):
                        total_score += 1
                    elif partial_final_answer_weight > 0:
                        total_score += partial_final_answer_weight
                elif partial_final_answer_weight > 0:
                    total_score += partial_final_answer_weight
                item["done"] = True
                continue

            tool_calls = extract_tool_calls_from_text(step_text)
            valid_tool_calls = _filter_valid_tool_calls(
                tool_calls, allowed_tool_names=None, fallback_tool_name=tool_name,
            )
            if not valid_tool_calls:
                item["done"] = True
                continue

            tool_call = valid_tool_calls[-1]
            use_code = python_code or tool_call.get("code", "")
            use_func = tool_call.get("function_name") or tool_call.get("name") or function_name
            tool_jobs.append({
                "state_idx": item_idx,
                "step_text": step_text,
                "use_code": use_code,
                "use_func": use_func,
                "call_args": _coerce_call_args(tool_call.get("arguments", {})),
                "timeout": tool_timeout,
                "tool_name": tool_call.get("name") or use_func or "tool",
                "expected_param_names": expected_param_names,
            })

        if not tool_jobs:
            continue

        if tool_eval_parallel and len(tool_jobs) > 1:
            max_workers = min(parallel_tool_workers, len(tool_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tool_results = list(executor.map(_execute_tool_job, tool_jobs))
        else:
            tool_results = [_execute_tool_job(job) for job in tool_jobs]

        tool_call_count += len(tool_results)
        for result in tool_results:
            item = items[result["state_idx"]]
            item["tool_called"] = True
            if result.get("reward_delta", 0) > 0:
                item["tool_success"] = True
                tool_success_call_count += 1
            else:
                tool_error_call_count += 1
            item["messages"].append({"role": "assistant", "content": result["step_text"]})
            item["messages"].append({"role": "tool", "name": result.get("tool_name") or "tool",
                                     "content": result["tool_result"]})
            item["prompt"] = _apply_chat_template(
                processing_class, item["messages"], tools=openai_tools, add_generation_prompt=True,
            )

    if tool_metrics is not None:
        tool_metrics["tool_call_count"] = tool_call_count
        tool_metrics["tool_success_call_count"] = tool_success_call_count
        tool_metrics["tool_error_call_count"] = tool_error_call_count
        tool_metrics["tool_called_items"] = sum(1 for item in items if item.get("tool_called"))
        tool_metrics["tool_success_items"] = sum(1 for item in items if item.get("tool_success"))
        tool_metrics["too_long_prompt_count"] = too_long_count
        tool_metrics["total_items"] = len(items)

    return total_score


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def _trainer_has_judge(trainer) -> bool:
    """Return True if the trainer is configured to use judge rewards."""
    return (
        bool(getattr(trainer, "use_judge_reward", False))
        and bool(getattr(trainer, "judge_base_url", None))
    )


def _batch_judge_build_states(
        candidates: list[tuple[int, dict, list[dict]]],
        trainer,
    ) -> dict[int, float]:
    """Score build states for code quality using the fixed LLM judge (call_judge_once).

    The judge model (trainer.judge_model) is fixed for the entire training session —
    it scores code_correctness, code_clarity, schema_quality, and schema_code_alignment.
    It is never updated by _LoraJudgeSyncCallback.

    Args:
        candidates: list of (state_idx, state, test_questions) tuples.
        trainer:    The GRPO trainer (provides judge config attributes).

    Returns:
        Dict mapping state_idx → score in [0.0, 1.0].
    """
    # Fixed throughout training — never synced with LoRA.
    judge_base_url = getattr(trainer, "judge_base_url", "http://localhost:8089/v1")
    judge_model = getattr(trainer, "judge_model", "Qwen/Qwen3-4B-Instruct-2507")
    judge_api_key = getattr(trainer, "judge_api_key", "TEST_API")
    judge_max_workers = int(getattr(trainer, "judge_max_workers", 64))
    judge_timeout_s = int(getattr(trainer, "judge_timeout_s", 30))
    judge_max_questions = int(getattr(trainer, "judge_max_questions", 3))

    logger.debug(
        "judge batch (code quality): model=%s url=%s n_candidates=%d",
        judge_model, judge_base_url, len(candidates),
    )

    client = openai.OpenAI(base_url=judge_base_url, api_key=judge_api_key)

    scores: dict[int, float] = {}

    # Pre-filter: run check_schema_code_alignment (pure CPU/AST, no HTTP) before
    # submitting to the thread pool.  Fatal misalignments (syntax error or missing
    # function name) get their penalty scores assigned immediately without ever
    # making a judge HTTP request — saving latency and judge-server load.
    # Soft misalignments are passed through to the judge but flagged for post-scoring.
    to_judge: list[tuple[int, dict, list[dict]]] = []
    soft_halve: set[int] = set()  # state_idxs that need score *= 0.5 after judge

    for idx, state, questions in candidates:
        python_code = state.get("python_code") or ""
        openai_tools = state.get("openai_tools") or []
        aligned, reason = check_schema_code_alignment(python_code, openai_tools)
        if aligned:
            to_judge.append((idx, state, questions))
        else:
            logger.debug("Schema-code misalignment for state_idx=%d: %s", idx, reason)
            if reason.startswith("syntax error in code:"):
                # Unparseable code: strong negative signal — skip judge entirely.
                scores[idx] = -0.5
            elif "not found in code" in reason:
                # Schema function name absent from callable set: same fatal condition
                # as step1's schema_function_mismatch.  Zero score, skip judge.
                scores[idx] = 0.0
            else:
                # Softer structural divergence: still call the judge for a meaningful
                # quality signal, but halve the result afterward.
                to_judge.append((idx, state, questions))
                soft_halve.add(idx)

    def _judge_one(state_idx: int, state: dict, questions: list[dict]) -> tuple[int, float]:
        python_code = state.get("python_code") or ""
        openai_tools = state.get("openai_tools") or []
        task_type = state.get("task_type", "build")
        eval_questions = questions[:judge_max_questions] if judge_max_questions > 0 else questions
        scores_dict = call_judge_once(
            python_code=python_code,
            openai_tools=openai_tools,
            questions=eval_questions,
            task_type=task_type,
            client=client,
            model=judge_model,
            timeout_s=judge_timeout_s,
        )
        score = overall_quality_score(scores_dict)
        if state_idx in soft_halve:
            score *= 0.5
        return state_idx, score

    if to_judge:
        n_workers = min(judge_max_workers, len(to_judge))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_judge_one, idx, state, questions): idx
                for idx, state, questions in to_judge
            }
            for future in as_completed(futures):
                try:
                    state_idx, score = future.result()
                    scores[state_idx] = score
                except Exception as exc:
                    state_idx = futures[future]
                    logger.debug("Judge call failed for state_idx=%d: %s", state_idx, exc)
                    scores[state_idx] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Tool Pool
# ---------------------------------------------------------------------------

class ToolPool:
    """Thread-safe pool of generated tools, keyed by task category.

    Stores (python_code, openai_tools) pairs collected from successful build
    tasks and makes them available for use tasks of the same category so the
    model can call pre-built tools instead of writing them from scratch.
    """

    def __init__(self, max_per_task: int = 10) -> None:
        self._pool: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self.max_per_task = max_per_task

    def add(self, task_category: str, python_code: str, openai_tools: list[dict], quality: float = 0.0) -> None:
        """Add a tool to the pool; evicts the lowest-quality entry when the category is full."""
        with self._lock:
            bucket = self._pool.setdefault(task_category, [])
            if len(bucket) >= self.max_per_task:
                # Remove the entry with the lowest quality score rather than the oldest.
                worst_idx = min(range(len(bucket)), key=lambda i: bucket[i].get("quality", 0.0))
                bucket.pop(worst_idx)
            bucket.append({"python_code": python_code, "openai_tools": openai_tools, "quality": quality})

    def get(self, task_category: str, max_tools: int = 3) -> list[dict]:
        """Return up to *max_tools* entries for the category (most recent first)."""
        with self._lock:
            bucket = self._pool.get(task_category, [])
            return list(reversed(bucket[-max_tools:])) if bucket else []

    def get_distractors(self, exclude_category: str, max_tools: int = 2) -> list[dict]:
        """Return up to *max_tools* entries from categories other than *exclude_category*.

        One entry is sampled randomly from each eligible category so the
        distractors are diverse across task types.
        """
        with self._lock:
            other_cats = [k for k, v in self._pool.items() if k != exclude_category and v]
        if not other_cats:
            return []
        random.shuffle(other_cats)
        result: list[dict] = []
        for cat in other_cats:
            if len(result) >= max_tools:
                break
            with self._lock:
                bucket = self._pool.get(cat, [])
                if bucket:
                    result.append(random.choice(bucket))
        return result

    def total_tools(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pool.values())

    def category_counts(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._pool.items()}


def _task_category_from_meta(meta: dict) -> str:
    """Extract task category from row metadata.

    The HF dataset rows have 'id' like 'bitwise_arithmetic-train-use-42'.
    task_name is computed in train_decompose_grpo.py but is NOT embedded in
    the prompt metadata — only the raw HF row is.  Derive it from 'id'.
    """
    row_id = (meta or {}).get("id", "")
    if row_id:
        return row_id.split("-train-")[0]
    return "unknown"


def _get_tool_pool(trainer) -> "ToolPool":
    """Lazily initialise and return the shared ToolPool on the trainer."""
    if not hasattr(trainer, "_tool_pool") or trainer._tool_pool is None:
        max_per_task = int(getattr(trainer, "tool_pool_max_per_task", 10))
        trainer._tool_pool = ToolPool(max_per_task=max_per_task)
    return trainer._tool_pool


def _build_pool_use_state(
        *,
        prompt_idx: int,
        meta: dict,
        target_entries: list[dict],
        distractor_entries: list[dict],
        trainer,
        rollout_batch_id: str,
        log_event,
    ) -> dict[str, Any]:
    """Build an initial state for a pool-covered use task (Phase 1 skipped).

    *target_entries* are the in-domain tools (correct category).
    *distractor_entries* come from unrelated categories and are mixed into the
    prompt so the model must identify the right tool.

    state["tool_name"] / state["function_name"] are always derived from
    *target_entries* so Phase 2 fallbacks are never a distractor name.

    Code is concatenated once per entry (not once per function inside it) to
    avoid duplicate function definitions in the combined script.
    """
    target_question = meta.get("target_question", "")
    task_category = _task_category_from_meta(meta)

    # Shuffle combined list so position in the tool list gives no hint.
    all_entries = target_entries + distractor_entries
    random.shuffle(all_entries)

    # Merge schemas and code: deduplicate by function name, add code once per entry.
    combined_code_parts: list[str] = []
    combined_tools: list[dict] = []
    seen_tool_names: set[str] = set()
    for entry in all_entries:
        pc = (entry.get("python_code") or "").strip()
        new_tools: list[dict] = []
        for tool in (entry.get("openai_tools") or []):
            tool_fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = tool_fn.get("name") if isinstance(tool_fn, dict) else None
            if name and name not in seen_tool_names:
                seen_tool_names.add(name)
                new_tools.append(tool)
        if new_tools:
            combined_tools.extend(new_tools)
            if pc:
                combined_code_parts.append(pc)  # once per entry, not per function

    combined_code: str | None = "\n\n".join(combined_code_parts) or None
    tool_names = list(seen_tool_names)

    # Derive primary tool_name / function_name from in-domain entries only
    # so the fallback in Phase 2 is never accidentally a distractor function.
    target_tool_names: list[str] = [
        t.get("function", {}).get("name")
        for e in target_entries
        for t in (e.get("openai_tools") or [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
        and t["function"].get("name")
    ]
    tool_name = target_tool_names[0] if target_tool_names else (tool_names[0] if tool_names else None)

    messages = [{"role": "user", "content": _format_user_question(target_question)}]
    phase2_prompt = _apply_chat_template(
        trainer.processing_class,
        messages,
        tools=combined_tools or None,
        add_generation_prompt=True,
    )

    # prompt_ids are a placeholder; they will be overwritten by Phase 2 generation.
    prompt_ids: list[int] = list(
        trainer.processing_class.encode(phase2_prompt, add_special_tokens=False)
    )

    state: dict[str, Any] = {
        "messages": messages,
        "prompt_ids": prompt_ids,
        "completion_ids": [],
        "logprobs": [],
        "env_mask": [],
        "env_reward": 0.0,
        "use_correct_reward": 0.0,
        "build_format_reward": 0.0,
        "num_turns": 0,
        "tool_exec_reward": 0.0,
        "final_answer_correct": False,
        "judge_reward": 0.0,
        "reward_components": {
            "step1_format_reward": 0.0,
            "schema_fn_reward": 0.0,
            "schema_param_reward": 0.0,
            "build_eval_reward": 0.0,
            "judge_reward": 0.0,
            "tool_exec_raw": 0.0,
            "tool_exec_norm": 0.0,
            "tool_exec_scaled": 0.0,
            "final_answer_reward": 0.0,
            "final_answer_correct": 0.0,
            "step1_failed": 0.0,
        },
        "step1_failure_reason": None,
        "python_code": combined_code,
        "openai_tools": combined_tools,
        "tool_name": tool_name,
        "tool_names": tool_names,
        "function_name": tool_name,
        "current_prompt_text": phase2_prompt,
        "done": False,
        "rollout_session_id": uuid.uuid4().hex,
        "prompt_index": prompt_idx,
        "task_type": "use",
        "target_answer": meta.get("target_answer"),
        "pool_used": True,
        "pool_tool_count": len(target_entries),
        "pool_distractor_count": len(distractor_entries),
    }

    if log_event is not None:
        log_event(
            "pool_use_task_ready",
            rollout_batch_id=rollout_batch_id,
            rollout_session_id=state["rollout_session_id"],
            prompt_index=prompt_idx,
            task_category=task_category,
            pool_tool_count=len(target_entries),
            pool_distractor_count=len(distractor_entries),
            target_tool_names=target_tool_names,
            all_tool_names=tool_names,
        )

    return state


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------

def decompose_rollout_w_judge(prompts: list[str], trainer) -> dict:
    """Decomposed rollout with LLM-judge quality rewards for build tasks."""
    from vllm import SamplingParams

    log_event = _get_rollout_logger(trainer)
    rollout_batch_id = uuid.uuid4().hex

    max_iters = getattr(trainer, "max_tool_calling_iterations", 5)
    tool_timeout = int(getattr(trainer, "tool_timeout", 30))
    step1_block_format_penalty = float(getattr(trainer, "step1_block_format_penalty", 0.5))
    parallel_tool_execution = bool(getattr(trainer, "parallel_tool_execution", True))
    default_workers = min(8, os.cpu_count() or 4)
    parallel_tool_workers = max(32, int(getattr(trainer, "parallel_tool_workers", default_workers)))

    tool_eval_max_questions = int(getattr(trainer, "tool_eval_max_questions", 8))
    tool_eval_parallel = bool(getattr(trainer, "tool_eval_parallel", True))

    phase2_max_tokens = int(getattr(trainer, "phase2_max_completion_length", 0) or 0)
    phase2_ratio = float(getattr(trainer, "phase2_max_completion_ratio", 0.25))
    if phase2_max_tokens <= 0:
        if phase2_ratio <= 0:
            phase2_ratio = 0.25
        phase2_max_tokens = max(64, int(trainer.max_completion_length * min(1.0, phase2_ratio)))

    use_judge = _trainer_has_judge(trainer)
    judge_reward_weight = float(getattr(trainer, "judge_reward_weight", 0.3))

    if log_event is not None:
        log_event(
            "rollout_batch_start",
            rollout_batch_id=rollout_batch_id,
            num_prompts=len(prompts),
            max_iters=max_iters,
            tool_timeout=tool_timeout,
            parallel_tool_execution=parallel_tool_execution,
            parallel_tool_workers=parallel_tool_workers,
            tool_eval_max_questions=tool_eval_max_questions,
            tool_eval_parallel=tool_eval_parallel,
            phase2_max_tokens=phase2_max_tokens,
            use_judge=use_judge,
            judge_reward_weight=judge_reward_weight,
        )

    llm_engine = _get_llm_engine(trainer)

    # Evaluator client for build task scoring (tool usability correctness).
    # The evaluator model is a lag-behind version of the training model, periodically
    # synced via _LoraJudgeSyncCallback. Read trainer.evaluator_model at call time
    # (not here) so each batch picks up the latest sync.
    _evaluator_base_url = getattr(trainer, "evaluator_base_url", "http://localhost:9002/v1")
    _evaluator_api_key = getattr(trainer, "evaluator_api_key", "TEST_API")
    evaluator_client = openai.OpenAI(base_url=_evaluator_base_url, api_key=_evaluator_api_key)

    cleaned_prompts = []
    metas = []
    for p in prompts:
        metadata, init_prompt = _strip_metadata(p)
        cleaned_prompts.append(init_prompt)
        metas.append(metadata)

    sampling_params = SamplingParams(
        n=1,
        temperature=trainer.temperature,
        top_p=trainer.top_p,
        top_k=trainer.top_k,
        max_tokens=trainer.max_completion_length,
        logprobs=1,
    )
    tool_sampling_params = SamplingParams(
        n=1,
        temperature=trainer.temperature,
        top_p=trainer.top_p,
        top_k=trainer.top_k,
        max_tokens=512,
        logprobs=1,
    )

    # ---- Tool pool: classify prompts ----
    tool_pool = _get_tool_pool(trainer)
    pool_max_per_task = int(getattr(trainer, "tool_pool_max_per_task", 20))
    pool_max_provide = int(getattr(trainer, "tool_pool_max_provide", 3))
    pool_distractor_count = int(getattr(trainer, "tool_pool_distractor_count", 2))
    tool_pool.max_per_task = pool_max_per_task

    # Decide which prompts need Phase 1 LLM generation.
    # Use tasks whose task category is already in the pool skip Phase 1.
    # For those tasks, also inject distractor tools from unrelated categories
    # so the model must identify the correct tool rather than calling blindly.
    phase1_prompt_indices: list[int] = []
    # Maps prompt_idx -> (target_entries, distractor_entries) kept separate
    # so _build_pool_use_state can anchor tool_name to in-domain functions.
    pool_use_entries: dict[int, tuple[list[dict], list[dict]]] = {}
    for _pi, _meta in enumerate(metas):
        _task_type_m = str((_meta or {}).get("task_type", "use")).lower()
        if _task_type_m == "use":
            _task_category = _task_category_from_meta(_meta)
            _entries = tool_pool.get(_task_category, max_tools=pool_max_provide)
            if _entries:
                _distractors = tool_pool.get_distractors(
                    _task_category, max_tools=pool_distractor_count
                )
                pool_use_entries[_pi] = (_entries, _distractors)
                continue
        phase1_prompt_indices.append(_pi)

    # Phase 1: tool creation (only for non-pool-covered prompts).
    if phase1_prompt_indices:
        phase1_outputs = llm_engine.generate(
            prompts=[cleaned_prompts[i] for i in phase1_prompt_indices],
            sampling_params=sampling_params,
        )
    else:
        phase1_outputs = []

    phase1_output_map: dict[int, Any] = {
        idx: out for idx, out in zip(phase1_prompt_indices, phase1_outputs)
    }

    states: list[dict[str, Any]] = []
    # Accumulate (state_idx, state, questions) for judge batch after Phase 1.
    judge_candidates: list[tuple[int, dict, list[dict]]] = []
    # Build states queued for background scoring (external evaluator, not llm_engine).
    _pending_build_score: list[tuple] = []

    for prompt_idx in range(len(prompts)):
        meta = metas[prompt_idx] or {}
        task_type = str(meta.get("task_type", "use")).lower()
        prompt_text = cleaned_prompts[prompt_idx]

        # Pool-covered use task: skip Phase 1 entirely.
        if prompt_idx in pool_use_entries:
            _target_entries, _distractor_entries = pool_use_entries[prompt_idx]
            states.append(_build_pool_use_state(
                prompt_idx=prompt_idx,
                meta=meta,
                target_entries=_target_entries,
                distractor_entries=_distractor_entries,
                trainer=trainer,
                rollout_batch_id=rollout_batch_id,
                log_event=log_event,
            ))
            continue

        initial_output = phase1_output_map[prompt_idx]

        messages = _parse_chatml_messages(prompt_text)
        initial_choice = initial_output.outputs[0]

        prompt_ids = initial_output.prompt_token_ids
        step1_ids = initial_choice.token_ids
        step1_logprobs = _extract_sampled_logprobs(initial_choice)
        step1_text = trainer.processing_class.decode(step1_ids, skip_special_tokens=False)

        env_mask = [1] * len(step1_ids)

        state: dict[str, Any] = {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "completion_ids": list(step1_ids),
            "logprobs": list(step1_logprobs) if step1_logprobs is not None else None,
            "env_mask": env_mask,
            "env_reward": 0.0,
            "use_correct_reward": 0.0,
            "build_format_reward": 0.0,
            "num_turns": 0,
            "tool_exec_reward": 0.0,
            "final_answer_correct": False,
            "judge_reward": 0.0,
            "reward_components": {
                "step1_format_reward": 0.0,
                "schema_fn_reward": 0.0,
                "schema_param_reward": 0.0,
                "build_eval_reward": 0.0,
                "judge_reward": 0.0,
                "tool_exec_raw": 0.0,
                "tool_exec_norm": 0.0,
                "tool_exec_scaled": 0.0,
                "final_answer_reward": 0.0,
                "final_answer_correct": 0.0,
                "step1_failed": 0.0,
            },
            "step1_failure_reason": None,
            "python_code": None,
            "openai_tools": [],
            "tool_name": None,
            "tool_names": [],
            "function_name": None,
            "current_prompt_text": "",
            "done": False,
            "rollout_session_id": uuid.uuid4().hex,
            "prompt_index": prompt_idx,
            "task_type": task_type,
            "target_answer": meta.get("target_answer"),
        }

        python_blocks = extract_code_blocks(step1_text, language="python")
        json_blocks = extract_code_blocks(step1_text, language="json")
        python_code = _remove_dunder_main_block(python_blocks[0]) if python_blocks else None
        json_block = json_blocks[0] if json_blocks else None
        valid_step1_block_format = _has_exactly_one_python_and_one_json_block(step1_text)

        step1_format_reward = abs(step1_block_format_penalty) if valid_step1_block_format else 0.0
        if step1_format_reward:
            state["env_reward"] += step1_format_reward
            state["build_format_reward"] += step1_format_reward
        state["reward_components"]["step1_format_reward"] = step1_format_reward

        openai_tools: list[dict] = []
        tool_name = None
        tool_names: list[str] = []
        function_name = None
        if json_block:
            try:
                raw_schema = json.loads(json_block)
                openai_tools = _normalize_tool_schema(raw_schema)
                if openai_tools:
                    tool_names = _extract_tool_names(openai_tools)
                    if tool_names:
                        tool_name = tool_names[0]
                        function_name = tool_name
            except (json.JSONDecodeError, TypeError, AttributeError):
                openai_tools = []

        state["python_code"] = python_code
        state["openai_tools"] = openai_tools
        state["tool_name"] = tool_name
        state["tool_names"] = tool_names
        state["function_name"] = function_name

        if log_event is not None:
            log_event(
                "step1_generated",
                rollout_batch_id=rollout_batch_id,
                rollout_session_id=state["rollout_session_id"],
                prompt_index=prompt_idx,
                task_type=task_type,
                prompt_preview=_safe_preview(prompt_text),
                prompt_token_count=len(prompt_ids),
                step1_completion_token_count=len(step1_ids),
                python_block_count=len(python_blocks),
                json_block_count=len(json_blocks),
                valid_step1_block_format=valid_step1_block_format,
                parsed_tool_name=tool_name,
                parsed_function_name=function_name,
                parsed_tools_count=len(openai_tools),
                parsed_tool_names=tool_names,
                step1_reward_after_penalty=state["env_reward"],
            )

        if not python_code or not openai_tools:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "missing_python_or_schema"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="missing_python_or_schema",
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        if not tool_names:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "schema_missing_function_name"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="schema_missing_function_name",
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        code_params_map = _extract_function_params_map(python_code)
        code_fn_names = _extract_function_names_from_code(python_code)
        missing_tools = [name for name in (tool_names or []) if name not in code_fn_names]
        if not missing_tools:
            state["env_reward"] += 0.5
            state["build_format_reward"] += 0.5
            state["reward_components"]["schema_fn_reward"] = 0.5
        else:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "schema_function_mismatch"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="schema_function_mismatch",
                    missing_tool_names=missing_tools,
                    code_function_names=code_fn_names,
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        schema_param_mismatches = []
        for tool in openai_tools:
            tool_fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            tool_name_check = tool_fn.get("name") if isinstance(tool_fn, dict) else None
            if not tool_name_check:
                continue
            schema_params = _extract_param_names_from_schema([tool])
            if not schema_params:
                continue
            code_params = code_params_map.get(tool_name_check, [])
            if not code_params:
                schema_param_mismatches.append({
                    "tool": tool_name_check,
                    "schema_params": schema_params,
                    "code_params": code_params,
                })
                continue
            if any(p not in code_params for p in schema_params):
                schema_param_mismatches.append({
                    "tool": tool_name_check,
                    "schema_params": schema_params,
                    "code_params": code_params,
                })

        if not schema_param_mismatches:
            state["env_reward"] += 0.5
            state["build_format_reward"] += 0.5
            state["reward_components"]["schema_param_reward"] = 0.5
        else:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "schema_param_mismatch"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="schema_param_mismatch",
                    schema_param_mismatches=schema_param_mismatches,
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        # Validate the OpenAI tool schema is structurally sound before sending to
        # the external evaluator.  A malformed schema (e.g. parameters.type != "object",
        # missing properties, invalid property types) causes the evaluator API to reject
        # or silently mishandle the call, resulting in zero tool calls and a spurious 0
        # build_eval_reward.  We catch it here and fail fast with a clear reason instead.
        schema_valid, schema_invalid_reason = validate_openai_tool_schema(openai_tools)
        if not schema_valid:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "invalid_tool_schema"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="invalid_tool_schema",
                    schema_invalid_reason=schema_invalid_reason,
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        # Check xgrammar compatibility of each tool's parameters schema.
        # When tool_choice="required" is used (forced on round 0 of the external
        # evaluator), vLLM compiles each tool's `parameters` schema with xgrammar
        # to constrain structured output.  Schemas that pass validate_openai_tool_schema
        # can still fail here (e.g. Draft-04 "items" as array, "anyOf", "uniqueItems").
        xgrammar_error: str | None = None
        for tool in openai_tools:
            params_schema = tool.get("function", {}).get("parameters", {})
            xgr_result = check_schema(params_schema)
            if not xgr_result.ok:
                tool_name = tool.get("function", {}).get("name", "<unknown>")
                xgrammar_error = f"tool '{tool_name}': {'; '.join(xgr_result.errors)}"
                break

        if xgrammar_error is not None:
            state["env_reward"] = 0
            state["build_format_reward"] = 0
            state["reward_components"]["step1_failed"] = 1.0
            state["step1_failure_reason"] = "xgrammar_incompatible_schema"
            state["done"] = True
            if log_event is not None:
                log_event(
                    "step1_failed",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=prompt_idx,
                    task_type=task_type,
                    reason="xgrammar_incompatible_schema",
                    schema_invalid_reason=xgrammar_error,
                    env_reward=state["env_reward"],
                )
            states.append(state)
            continue

        if task_type == "build":
            test_set = meta.get("tool_test_set", []) or []
            if tool_eval_max_questions > 0:
                test_set = test_set[:tool_eval_max_questions]

            tool_metrics: dict[str, Any] = {}
            state["build_tool_metrics"] = tool_metrics
            state["done"] = True

            # Queue for background scoring.  The external evaluator (_score_build_task
            # via evaluator_client) is pure HTTP — it never calls llm_engine — so it can
            # run concurrently with Phase 2 (llm_engine.generate for use tasks).
            # Scoring results are applied to the state after Phase 2 finishes (see below).
            _pending_build_score.append((len(states), state, test_set, len(test_set), tool_metrics, meta))

            # Collect for batch judge (also fired concurrently with Phase 2)
            if use_judge:
                judge_candidates.append((len(states), state, test_set))

            states.append(state)
            continue

        # =================================================== USE-TOOL-RL ================================================
        state["completion_ids"] = []
        state["env_mask"] = []
        state["logprobs"] = []

        target_question = meta.get("target_question")

        state["messages"] = [
            {"role": "user", "content": _format_user_question(target_question)},
        ]

        state["current_prompt_text"] = _apply_chat_template(
            trainer.processing_class,
            state["messages"],
            tools=openai_tools or None,
            add_generation_prompt=True,
        )

        _maybe_print_messages(trainer, state, header="TOOL-USE PROMPT (STEP2 START)")

        states.append(state)

    # ---------------------------------------------------------------------------
    # Fire build-scoring and judge requests in background threads, then run
    # Phase 2 (llm_engine.generate for use tasks) concurrently with both.
    #
    # _score_build_task uses evaluator_client (external HTTP, never llm_engine).
    # _batch_judge_build_states also uses an external HTTP judge service.
    # Phase 2 is the only part that calls llm_engine, so all three workloads
    # are fully independent and can overlap safely.
    # ---------------------------------------------------------------------------
    _evaluator_model_snapshot = getattr(trainer, "evaluator_model", None)

    # Cap concurrent build-scoring workers to avoid two problems:
    #   (1) Thread explosion: without a cap, _pending_build_score can reach ~170
    #       entries.  Each worker spawns a nested ThreadPoolExecutor(16) inside
    #       _score_build_task → 170 * 16 ≈ 2 720 extra threads per rollout.
    #   (2) CPU starvation of vLLM: each worker may launch a subprocess to execute
    #       generated tool code.  With 64 concurrent subprocesses on a 24-core host,
    #       vLLM's scheduling loop is starved of CPU time → Phase-2 generation drops
    #       from ~37 tok/s to ~10 tok/s.  Limiting workers to ≈1/3 of CPU cores
    #       leaves enough cores for vLLM while still parallelising build scoring.
    # Fix: (1) cap the outer executor; (2) disable per-question inner parallelism
    # inside each thread (tool_eval_parallel=False) because the outer pool already
    # provides parallelism across build states.
    import os as _os_build
    _cpu_cap = max(4, (_os_build.cpu_count() or 8) // 3)  # ~1/3 of cores, min 4
    _MAX_BUILD_SCORER_WORKERS = min(
        max(1, len(_pending_build_score)),
        _cpu_cap,
    )

    def _run_one_build_score(entry: tuple) -> tuple:
        _sidx, _state, _test_set, _test_set_len, _tool_metrics, _meta = entry
        try:
            # Allow limited inner parallelism: evaluate questions concurrently but
            # cap at 4 workers so total threads stay at _MAX_BUILD_SCORER_WORKERS * 4.
            # This reduces per-future wall time from (questions * rounds * latency)
            # to (ceil(questions/4) * rounds * latency) — a 4x speedup per build state.
            _inner_workers = min(4, len(_test_set) if _test_set else 1)
            _total_score = _score_build_task(
                openai_tools=_state["openai_tools"],
                python_code=_state["python_code"],
                tool_timeout=tool_timeout,
                test_set=_test_set,
                tool_eval_parallel=True,
                parallel_tool_workers=_inner_workers,
                tool_metrics=_tool_metrics,
                evaluator_client=evaluator_client,
                evaluator_model=_evaluator_model_snapshot,
            )
        except Exception as _exc:
            logger.debug("Background build scoring failed for state_idx=%d: %s", _sidx, _exc)
            _total_score = 0
        return _sidx, _total_score, _test_set_len, _meta

    _build_executor = ThreadPoolExecutor(max_workers=_MAX_BUILD_SCORER_WORKERS)
    _build_score_futures = [
        _build_executor.submit(_run_one_build_score, entry)
        for entry in _pending_build_score
    ]

    _judge_executor = None
    _judge_future = None
    if use_judge and judge_candidates:
        if log_event is not None:
            log_event(
                "judge_batch_start",
                rollout_batch_id=rollout_batch_id,
                num_candidates=len(judge_candidates),
            )
        _judge_executor = ThreadPoolExecutor(max_workers=1)
        _judge_future = _judge_executor.submit(_batch_judge_build_states, judge_candidates, trainer)

    # Phase 2: tool use for Task-2 states (concurrent with build scoring + judge above).
    use_state_idxs = [idx for idx, state in enumerate(states) if state["task_type"] == "use" and not state["done"]]
    if use_state_idxs:
        step2_outputs = llm_engine.generate(
            prompts=[states[idx]["current_prompt_text"] for idx in use_state_idxs],
            sampling_params=tool_sampling_params,
        )

        tool_jobs: list[dict[str, Any]] = []
        for state_idx, current_output in zip(use_state_idxs, step2_outputs, strict=False):
            state = states[state_idx]
            step_choice = current_output.outputs[0]
            step_ids = step_choice.token_ids
            step_logprobs = _extract_sampled_logprobs(step_choice)
            step_text = trainer.processing_class.decode(step_ids, skip_special_tokens=False)

            state["prompt_ids"] = current_output.prompt_token_ids
            state["completion_ids"] = list(step_ids)
            state["env_mask"] = [1] * len(step_ids)
            state["logprobs"] = list(step_logprobs) if step_logprobs is not None else None

            final_answer = _extract_final_answer(step_text)
            if final_answer is not None:
                is_correct = answers_match(final_answer, state.get("target_answer"))
                state["final_answer_correct"] = bool(is_correct)
                state["final_answer_text"] = final_answer  # stored for judge fallback
                state["done"] = True
                if log_event is not None:
                    log_event(
                        "phase2_final_answer_terminate",
                        rollout_batch_id=rollout_batch_id,
                        rollout_session_id=state["rollout_session_id"],
                        prompt_index=state["prompt_index"],
                        turn_idx=0,
                        final_answer=final_answer,
                        is_correct=is_correct,
                    )
                continue

            tool_calls = extract_tool_calls_from_text(step_text)
            allowed_tool_names = set(state.get("tool_names") or [])
            valid_tool_calls = _filter_valid_tool_calls(
                tool_calls,
                allowed_tool_names=allowed_tool_names,
                fallback_tool_name=state.get("tool_name"),
            )

            if not valid_tool_calls:
                state["done"] = True
                if log_event is not None:
                    log_event(
                        "phase2_no_tool_call_terminate",
                        rollout_batch_id=rollout_batch_id,
                        rollout_session_id=state["rollout_session_id"],
                        prompt_index=state["prompt_index"],
                        turn_idx=0,
                        step_text_preview=_safe_preview(step_text),
                    )
                continue

            tool_call = valid_tool_calls[-1]
            call_args = _coerce_call_args(tool_call.get("arguments", {}))

            use_code = state["python_code"] or tool_call.get("code", "")
            use_func = tool_call.get("function_name") or tool_call.get("name")
            if not use_func:
                state["done"] = True
                continue
            state["num_turns"] += 1

            if log_event is not None:
                log_event(
                    "phase2_tool_call_detected",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=state["prompt_index"],
                    turn_idx=0,
                    tool_name=tool_call.get("name"),
                    use_func=use_func,
                    has_inline_code=bool(tool_call.get("code")),
                    arguments=call_args,
                )

            _called_name = use_func
            _matching_tools = [
                t for t in (state["openai_tools"] or [])
                if isinstance(t, dict)
                and isinstance(t.get("function"), dict)
                and t["function"].get("name") == _called_name
            ]
            tool_jobs.append(
                {
                    "state_idx": state_idx,
                    "step_text": step_text,
                    "use_code": use_code,
                    "use_func": use_func,
                    "call_args": call_args,
                    "timeout": tool_timeout,
                    "tool_name": _called_name,
                    "expected_param_names": (
                        _extract_param_names_from_schema(_matching_tools)
                        or _extract_function_params_map(use_code).get(_called_name)
                    ),
                }
            )

        if tool_jobs:
            if parallel_tool_execution and len(tool_jobs) > 1:
                max_workers = min(parallel_tool_workers, len(tool_jobs))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    tool_results = list(executor.map(_execute_tool_job, tool_jobs))
            else:
                tool_results = [_execute_tool_job(job) for job in tool_jobs]

            for result in tool_results:
                state = states[result["state_idx"]]
                step_text = result["step_text"]
                tool_result = result["tool_result"]
                state["tool_exec_reward"] += result["reward_delta"]
                state["env_reward"] += result["reward_delta"]
                state["reward_components"]["tool_exec_raw"] += result["reward_delta"]

                if log_event is not None:
                    log_event(
                        "phase2_tool_execution_result",
                        rollout_batch_id=rollout_batch_id,
                        rollout_session_id=state["rollout_session_id"],
                        prompt_index=state["prompt_index"],
                        turn_idx=0,
                        reward_delta=result["reward_delta"],
                        env_reward=state["env_reward"],
                        tool_result_preview=_safe_preview(tool_result),
                        step_text_preview=_safe_preview(step_text),
                    )

                if state["messages"] is not None:
                    messages = state["messages"]
                    messages.append({"role": "assistant", "content": step_text})
                    before_tool_text = _apply_chat_template(
                        trainer.processing_class,
                        messages,
                        tools=state["openai_tools"] or None,
                        add_generation_prompt=False,
                    )
                    tool_role_name = result.get("tool_name") or state["tool_name"] or "tool"
                    messages.append({"role": "tool", "name": tool_role_name, "content": tool_result})
                    after_tool_text = _apply_chat_template(
                        trainer.processing_class,
                        messages,
                        tools=state["openai_tools"] or None,
                        add_generation_prompt=False,
                    )
                    if after_tool_text.startswith(before_tool_text):
                        tool_result_text = after_tool_text[len(before_tool_text):]
                    else:
                        tool_result_text = f"\n\nTool result: {tool_result}\n\n"

                    tool_result_ids = trainer.processing_class.encode(tool_result_text, add_special_tokens=False)
                    state["completion_ids"].extend(tool_result_ids)
                    state["env_mask"].extend([0] * len(tool_result_ids))
                    if state["logprobs"] is not None:
                        state["logprobs"].extend([0.0] * len(tool_result_ids))
                    state["current_prompt_text"] = _apply_chat_template(
                        trainer.processing_class,
                        messages,
                        tools=state["openai_tools"] or None,
                        add_generation_prompt=True,
                    )
                    _maybe_print_messages(trainer, state, header="TOOL-USE AFTER TOOL RESULT")
                else:
                    tool_result_text = f"\n\nTool result: {tool_result}\n\n"
                    tool_result_ids = trainer.processing_class.encode(tool_result_text, add_special_tokens=False)
                    state["completion_ids"].extend(tool_result_ids)
                    state["env_mask"].extend([0] * len(tool_result_ids))
                    if state["logprobs"] is not None:
                        state["logprobs"].extend([0.0] * len(tool_result_ids))
                    state["current_prompt_text"] = state["current_prompt_text"] + step_text + tool_result_text

    # Additional tool-call turns (turn_idx >= 1)
    for turn_idx in range(1, max_iters):
        active_state_idxs = [idx for idx, state in enumerate(states) if not state["done"]]
        if not active_state_idxs:
            break

        step_outputs = llm_engine.generate(
            prompts=[states[idx]["current_prompt_text"] for idx in active_state_idxs],
            sampling_params=tool_sampling_params,
        )

        tool_jobs: list[dict[str, Any]] = []
        for state_idx, current_output in zip(active_state_idxs, step_outputs, strict=False):
            state = states[state_idx]
            step_choice = current_output.outputs[0]
            step_ids = step_choice.token_ids
            step_logprobs = _extract_sampled_logprobs(step_choice)
            step_text = trainer.processing_class.decode(step_ids, skip_special_tokens=False)

            prev_len = len(state["completion_ids"])
            state["completion_ids"].extend(step_ids)
            state["env_mask"].extend([1] * len(step_ids))
            if step_logprobs is not None:
                if state["logprobs"] is None:
                    state["logprobs"] = [0.0] * prev_len + step_logprobs
                else:
                    state["logprobs"].extend(step_logprobs)
            elif state["logprobs"] is not None:
                state["logprobs"].extend([0.0] * len(step_ids))

            final_answer = _extract_final_answer(step_text)
            if final_answer is not None:
                is_correct = answers_match(final_answer, state.get("target_answer"))
                state["final_answer_correct"] = bool(is_correct)
                state["final_answer_text"] = final_answer  # stored for judge fallback
                state["done"] = True
                if log_event is not None:
                    log_event(
                        "phase2_final_answer_terminate",
                        rollout_batch_id=rollout_batch_id,
                        rollout_session_id=state["rollout_session_id"],
                        prompt_index=state["prompt_index"],
                        turn_idx=turn_idx,
                        final_answer=final_answer,
                        is_correct=is_correct,
                    )
                continue

            tool_calls = extract_tool_calls_from_text(step_text)
            allowed_tool_names = set(state.get("tool_names") or [])
            valid_tool_calls = _filter_valid_tool_calls(
                tool_calls,
                allowed_tool_names=allowed_tool_names,
                fallback_tool_name=state.get("tool_name"),
            )

            if not valid_tool_calls:
                state["done"] = True
                if log_event is not None:
                    log_event(
                        "phase2_no_tool_call_terminate",
                        rollout_batch_id=rollout_batch_id,
                        rollout_session_id=state["rollout_session_id"],
                        prompt_index=state["prompt_index"],
                        turn_idx=turn_idx,
                        step_text_preview=_safe_preview(step_text),
                    )
                continue

            tool_call = valid_tool_calls[-1]
            call_args = _coerce_call_args(tool_call.get("arguments", {}))

            use_code = state["python_code"] or tool_call.get("code", "")
            use_func = tool_call.get("function_name") or tool_call.get("name")
            if not use_func:
                state["done"] = True
                continue
            state["num_turns"] += 1

            if log_event is not None:
                log_event(
                    "phase2_tool_call_detected",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=state["prompt_index"],
                    turn_idx=turn_idx,
                    tool_name=tool_call.get("name"),
                    use_func=use_func,
                    has_inline_code=bool(tool_call.get("code")),
                    arguments=call_args,
                )

            _called_name = use_func
            _matching_tools = [
                t for t in (state["openai_tools"] or [])
                if isinstance(t, dict)
                and isinstance(t.get("function"), dict)
                and t["function"].get("name") == _called_name
            ]
            tool_jobs.append(
                {
                    "state_idx": state_idx,
                    "step_text": step_text,
                    "use_code": use_code,
                    "use_func": use_func,
                    "call_args": call_args,
                    "timeout": tool_timeout,
                    "tool_name": _called_name,
                    "expected_param_names": (
                        _extract_param_names_from_schema(_matching_tools)
                        or _extract_function_params_map(use_code).get(_called_name)
                    ),
                }
            )

        if not tool_jobs:
            continue

        if parallel_tool_execution and len(tool_jobs) > 1:
            max_workers = min(parallel_tool_workers, len(tool_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tool_results = list(executor.map(_execute_tool_job, tool_jobs))
        else:
            tool_results = [_execute_tool_job(job) for job in tool_jobs]

        for result in tool_results:
            state = states[result["state_idx"]]
            step_text = result["step_text"]
            tool_result = result["tool_result"]
            state["tool_exec_reward"] += result["reward_delta"]
            state["env_reward"] += result["reward_delta"]
            state["reward_components"]["tool_exec_raw"] += result["reward_delta"]

            if log_event is not None:
                log_event(
                    "phase2_tool_execution_result",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=state["prompt_index"],
                    turn_idx=turn_idx,
                    reward_delta=result["reward_delta"],
                    env_reward=state["env_reward"],
                    tool_result_preview=_safe_preview(tool_result),
                    step_text_preview=_safe_preview(step_text),
                )

            if state["messages"] is not None:
                messages = state["messages"]
                messages.append({"role": "assistant", "content": step_text})
                before_tool_text = _apply_chat_template(
                    trainer.processing_class,
                    messages,
                    tools=state["openai_tools"] or None,
                    add_generation_prompt=False,
                )
                tool_role_name = result.get("tool_name") or state["tool_name"] or "tool"
                messages.append({"role": "tool", "name": tool_role_name, "content": tool_result})
                after_tool_text = _apply_chat_template(
                    trainer.processing_class,
                    messages,
                    tools=state["openai_tools"] or None,
                    add_generation_prompt=False,
                )
                if after_tool_text.startswith(before_tool_text):
                    tool_result_text = after_tool_text[len(before_tool_text):]
                else:
                    tool_result_text = f"\n\nTool result: {tool_result}\n\n"

                tool_result_ids = trainer.processing_class.encode(tool_result_text, add_special_tokens=False)
                state["completion_ids"].extend(tool_result_ids)
                state["env_mask"].extend([0] * len(tool_result_ids))
                if state["logprobs"] is not None:
                    state["logprobs"].extend([0.0] * len(tool_result_ids))
                state["current_prompt_text"] = _apply_chat_template(
                    trainer.processing_class,
                    messages,
                    tools=state["openai_tools"] or None,
                    add_generation_prompt=True,
                )
                _maybe_print_messages(trainer, state, header="TOOL-USE AFTER TOOL RESULT (TURN)")
            else:
                tool_result_text = f"\n\nTool result: {tool_result}\n\n"
                tool_result_ids = trainer.processing_class.encode(tool_result_text, add_special_tokens=False)
                state["completion_ids"].extend(tool_result_ids)
                state["env_mask"].extend([0] * len(tool_result_ids))
                if state["logprobs"] is not None:
                    state["logprobs"].extend([0.0] * len(tool_result_ids))
                state["current_prompt_text"] = state["current_prompt_text"] + step_text + tool_result_text

    # ---------------------------------------------------------------------------
    # Collect build-scoring and build-judge futures (submitted before Phase 2)
    # ---------------------------------------------------------------------------
    # Per-future deadline: tool_timeout * max_rounds * num_test_questions + headroom.
    # This prevents a single slow evaluator request from blocking the entire batch
    # indefinitely (default openai timeout is 600 s; each build state can run
    # multiple rounds over multiple questions).
    #
    # IMPORTANT: the timeout must be on as_completed() itself, NOT on .result().
    # as_completed() only yields futures that are already done, so .result() always
    # returns immediately — a timeout there is useless. The iterator is what blocks.
    _build_future_timeout = max(tool_timeout * max_iters * tool_eval_max_questions + 60, 40)
    from concurrent.futures import TimeoutError as _FuturesTimeoutError
    try:
      for _bfuture in as_completed(_build_score_futures, timeout=_build_future_timeout):
        try:
            _bidx, _btotal_score, _btest_set_len, _bmeta = _bfuture.result()
        except Exception as _bexc:
            logger.debug("Build scoring future raised: %s", _bexc)
            continue
        _bstate = states[_bidx]
        _btool_metrics = _bstate["build_tool_metrics"]
        _btotal_eval_items = _btool_metrics.get("total_items", _btest_set_len)
        _bbuild_eval_reward = _btotal_score / _btotal_eval_items if _btotal_eval_items > 0 else 0.0
        _bstate["reward_components"]["build_eval_reward"] = _bbuild_eval_reward
        _bstate["env_reward"] += _bbuild_eval_reward

        if _btotal_score > 0 and _bstate["python_code"] and _bstate["openai_tools"]:
            _btask_category = _task_category_from_meta(_bmeta)
            tool_pool.add(_btask_category, _bstate["python_code"], _bstate["openai_tools"], quality=_bbuild_eval_reward)
            if log_event is not None:
                log_event(
                    "tool_pool_add",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=_bstate["rollout_session_id"],
                    task_category=_btask_category,
                    build_score=_btotal_score,
                    pool_total_tools=tool_pool.total_tools(),
                )

        if log_event is not None:
            log_event(
                "build_task_scored",
                rollout_batch_id=rollout_batch_id,
                rollout_session_id=_bstate["rollout_session_id"],
                prompt_index=_bstate["prompt_index"],
                score=_btotal_score,
                env_reward=_bstate["env_reward"],
                test_count=_btest_set_len,
                tool_call_count=_btool_metrics.get("tool_call_count", 0),
                tool_success_call_count=_btool_metrics.get("tool_success_call_count", 0),
                tool_error_call_count=_btool_metrics.get("tool_error_call_count", 0),
                tool_called_items=_btool_metrics.get("tool_called_items", 0),
                tool_success_items=_btool_metrics.get("tool_success_items", 0),
            )
    except _FuturesTimeoutError:
        logger.warning(
            "Build scoring timed out after %ds; incomplete futures will be skipped.",
            _build_future_timeout,
        )

    _build_executor.shutdown(wait=False)

    if _judge_future is not None:
        _judge_timeout = int(getattr(trainer, "judge_timeout_s", 30)) * int(getattr(trainer, "judge_max_questions", 3)) + 60
        try:
            _judge_scores = _judge_future.result(timeout=_judge_timeout)
        except Exception as _jexc:
            logger.warning("Judge future raised or timed out after %ds: %s", _judge_timeout, _jexc)
            _judge_scores = {}
        if _judge_executor is not None:
            _judge_executor.shutdown(wait=False)
        for state_idx, score in _judge_scores.items():
            state = states[state_idx]
            state["judge_reward"] = score
            state["reward_components"]["judge_reward"] = score

            if log_event is not None:
                log_event(
                    "judge_scored",
                    rollout_batch_id=rollout_batch_id,
                    rollout_session_id=state["rollout_session_id"],
                    prompt_index=state["prompt_index"],
                    judge_score=score,
                    judge_reward_weighted=score * judge_reward_weight,
                    env_reward_after_judge=state["env_reward"],
                )

    # Judge fallback: for use states where answers_match failed but the model
    # did produce a final answer, ask the LLM judge whether the two expressions
    # are equivalent (handles hex, fractions, symbolic forms, etc.).
    if _trainer_has_judge(trainer):
        _judge_fallback_candidates = [
            st for st in states
            if st.get("task_type") == "use"
            and not st.get("final_answer_correct")
            and st.get("final_answer_text")
            and st.get("target_answer")
        ]
        if _judge_fallback_candidates:
            from tool_rl.llm_judge_eq import llm_as_judge_equal
            _judge_base_url = getattr(trainer, "judge_base_url", "http://localhost:8089/v1")
            _judge_api_key = getattr(trainer, "judge_api_key", "TEST_API")
            _judge_model = getattr(trainer, "judge_model", None)
            _judge_timeout = int(getattr(trainer, "judge_timeout_s", 30))
            _judge_max_workers = int(getattr(trainer, "judge_max_workers", 64))

            def _eq_judge_one(st):
                try:
                    return st, llm_as_judge_equal(
                        st["final_answer_text"],
                        str(st["target_answer"]),
                        model=_judge_model,
                        base_url=_judge_base_url,
                        api_key=_judge_api_key,
                        timeout=_judge_timeout,
                    )
                except Exception:
                    return st, False

            with ThreadPoolExecutor(max_workers=min(_judge_max_workers, len(_judge_fallback_candidates))) as _pool:
                for _st, _eq in _pool.map(_eq_judge_one, _judge_fallback_candidates):
                    if _eq:
                        _st["final_answer_correct"] = True

    for state in states:
        if state.get("task_type") == "use":
            correct = 1.0 if state.get("final_answer_correct") else 0.0
            max_tool_exec = TOOL_EXEC_SUCCESS_REWARD * max_iters if max_iters > 0 else 0.0
            tool_exec_norm = 0.0
            if max_tool_exec > 0:
                tool_exec_norm = (state.get("tool_exec_reward") or 0.0) / max_tool_exec
                if tool_exec_norm > 1.0:
                    tool_exec_norm = 1.0

            tool_exec_scaled = tool_exec_norm

            num_turns = state.get("num_turns") or 0
            turn_ratio = min(num_turns / max_iters, 1.0) if max_iters > 0 else 0.0
            if turn_ratio <= 0.5:
                efficiency_multiplier = 1.0 - 0.3 * (turn_ratio * 2)
            else:
                t = (turn_ratio - 0.5) * 2
                efficiency_multiplier = 0.7 * (1.0 - t) ** 2

            correct_reward = 2.0 * correct * efficiency_multiplier

            # Track a small partial-attempt signal for diagnostics, but keep it
            # out of env_reward so reward axes stay disentangled for GDPO.
            _partial_weight = float(getattr(trainer, "partial_final_answer_reward", 0.1))
            final_attempt_reward = (
                _partial_weight
                if state.get("final_answer_text") is not None and not correct
                else 0.0
            )

            # Reward shaping: penalise rollouts where the model never attempted
            # a final answer.  This teaches the model that exhausting tool-call
            # turns without answering is worse than trying to reason directly.
            # The penalty is kept small so it does not swamp the tool-exec signal.
            _no_answer_penalty_weight = float(getattr(trainer, "no_answer_penalty", 0.5))
            no_answer_penalty = (
                -_no_answer_penalty_weight
                if state.get("final_answer_text") is None
                else 0.0
            )

            state["use_correct_reward"] = correct_reward
            state["env_reward"] = tool_exec_scaled + no_answer_penalty

            if getattr(trainer, "debug_use_rewards", True):
                _debug_path = getattr(trainer, "debug_use_rewards_path", "/mnt/disk/toolmakers/tool_debug.md")
                _line = (
                    f"| {state.get('prompt_index')} "
                    f"| {state.get('pool_used', False)} "
                    f"| {state.get('target_answer')!r} "
                    f"| {state.get('final_answer_text')!r} "
                    f"| {bool(state.get('final_answer_correct'))} "
                    f"| {state.get('tool_exec_reward', 0.0):.3f} "
                    f"| {state.get('num_turns', 0)} "
                    f"| {efficiency_multiplier:.3f} "
                    f"| {correct_reward:.3f} "
                    f"| {tool_exec_scaled:.3f} "
                    f"| {final_attempt_reward:.3f} "
                    f"| {state['env_reward']:.3f} |\n"
                )
                try:
                    with open(_debug_path, "a", encoding="utf-8") as _f:
                        _f.write(_line)
                except OSError:
                    pass

            if isinstance(state.get("reward_components"), dict):
                state["reward_components"]["tool_exec_raw"] = float(state.get("tool_exec_reward") or 0.0)
                state["reward_components"]["tool_exec_norm"] = tool_exec_norm
                state["reward_components"]["tool_exec_scaled"] = tool_exec_scaled
                state["reward_components"]["efficiency_multiplier"] = efficiency_multiplier
                state["reward_components"]["final_answer_reward"] = correct_reward
                state["reward_components"]["final_answer_correct"] = correct
                state["reward_components"]["final_attempt_reward"] = final_attempt_reward
                state["reward_components"]["no_answer_penalty"] = no_answer_penalty

    if log_event is not None:
        for state in states:
            log_event(
                "rollout_prompt_end",
                rollout_batch_id=rollout_batch_id,
                rollout_session_id=state["rollout_session_id"],
                prompt_index=state["prompt_index"],
                task_type=state["task_type"],
                num_turns=state["num_turns"],
                env_reward=state["env_reward"],
                judge_reward=state.get("judge_reward", 0.0),
                prompt_token_count=len(state["prompt_ids"]),
                completion_token_count=len(state["completion_ids"]),
                had_messages=state["messages"] is not None,
            )

    prompt_ids_list = [state["prompt_ids"] for state in states]
    completion_ids_list = [state["completion_ids"] for state in states]
    logprobs_list = [state["logprobs"] for state in states]
    env_mask_list = [state["env_mask"] for state in states]
    env_reward_list = [state["env_reward"] for state in states]
    use_correct_reward_list = [state.get("use_correct_reward", 0.0) for state in states]
    num_turns_list = [state["num_turns"] for state in states]
    task_type_list = [state.get("task_type") for state in states]
    judge_reward_list = [state.get("judge_reward", 0.0) for state in states]

    if num_turns_list and hasattr(trainer, "_metrics"):
        mode = "train" if trainer.model.training else "eval"
        trainer._metrics[mode]["num_turns_mean"].append(sum(num_turns_list) / len(num_turns_list))
        trainer._metrics[mode]["num_turns_min"].append(min(num_turns_list))
        trainer._metrics[mode]["num_turns_max"].append(max(num_turns_list))
        build_states = [s for s in states if s.get("task_type") == "build"]
        use_states = [s for s in states if s.get("task_type") == "use"]

        def _append_mean(metric: str, values: list[float]):
            if values:
                trainer._metrics[mode][metric].append(sum(values) / len(values))

        def _collect_component(src_states: list[dict[str, Any]], key: str) -> list[float]:
            vals: list[float] = []
            for st in src_states:
                if key == "env_reward":
                    vals.append(float(st.get("env_reward") or 0.0))
                    continue
                if key == "build_format_reward":
                    vals.append(float(st.get("build_format_reward") or 0.0))
                    continue
                if key == "judge_reward":
                    vals.append(float(st.get("judge_reward") or 0.0))
                    continue
                comp = st.get("reward_components") or {}
                try:
                    vals.append(float(comp.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    vals.append(0.0)
            return vals

        def _append_rate(metric: str, matches: int, total: int):
            if total > 0:
                trainer._metrics[mode][metric].append(matches / total)

        def _collect_build_tool_metric(key: str) -> list[float]:
            vals: list[float] = []
            for st in build_states:
                metrics = st.get("build_tool_metrics") or {}
                try:
                    vals.append(float(metrics.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    vals.append(0.0)
            return vals

        def _collect_build_tool_success_rate() -> list[float]:
            rates: list[float] = []
            for st in build_states:
                metrics = st.get("build_tool_metrics") or {}
                try:
                    total = float(metrics.get("tool_call_count", 0.0) or 0.0)
                    success = float(metrics.get("tool_success_call_count", 0.0) or 0.0)
                except (TypeError, ValueError):
                    total = 0.0
                    success = 0.0
                if total > 0:
                    rates.append(success / total)
                else:
                    rates.append(0.0)
            return rates

        if build_states:
            _append_mean("rollout/build/env_reward", _collect_component(build_states, "env_reward"))
            _append_mean("rollout/build/build_format_reward", _collect_component(build_states, "build_format_reward"))
            _append_mean("rollout/build/step1_format_reward", _collect_component(build_states, "step1_format_reward"))
            _append_mean("rollout/build/schema_fn_reward", _collect_component(build_states, "schema_fn_reward"))
            _append_mean("rollout/build/schema_param_reward", _collect_component(build_states, "schema_param_reward"))
            _append_mean("rollout/build/build_eval_reward", _collect_component(build_states, "build_eval_reward"))
            _append_mean("rollout/build/judge_reward", _collect_component(build_states, "judge_reward"))
            _append_mean("rollout/build/tool_call_count", _collect_build_tool_metric("tool_call_count"))
            _append_mean("rollout/build/tool_success_call_count", _collect_build_tool_metric("tool_success_call_count"))
            _append_mean("rollout/build/tool_error_call_count", _collect_build_tool_metric("tool_error_call_count"))
            _append_mean("rollout/build/tool_success_call_rate", _collect_build_tool_success_rate())
            _append_rate(
                "rollout/build/step1_failed_rate",
                sum(1 for st in build_states if st.get("step1_failure_reason")),
                len(build_states),
            )
            _append_rate(
                "rollout/build/step1_fail_missing_python_or_schema_rate",
                sum(1 for st in build_states if st.get("step1_failure_reason") == "missing_python_or_schema"),
                len(build_states),
            )
            _append_rate(
                "rollout/build/step1_fail_schema_missing_function_name_rate",
                sum(1 for st in build_states if st.get("step1_failure_reason") == "schema_missing_function_name"),
                len(build_states),
            )
            _append_rate(
                "rollout/build/step1_fail_schema_function_mismatch_rate",
                sum(1 for st in build_states if st.get("step1_failure_reason") == "schema_function_mismatch"),
                len(build_states),
            )
            _append_rate(
                "rollout/build/step1_fail_schema_param_mismatch_rate",
                sum(1 for st in build_states if st.get("step1_failure_reason") == "schema_param_mismatch"),
                len(build_states),
            )

        if use_states:
            _append_mean("rollout/use/env_reward", _collect_component(use_states, "env_reward"))
            _append_mean("rollout/use/build_format_reward", _collect_component(use_states, "build_format_reward"))
            _append_mean("rollout/use/step1_format_reward", _collect_component(use_states, "step1_format_reward"))
            _append_mean("rollout/use/schema_fn_reward", _collect_component(use_states, "schema_fn_reward"))
            _append_mean("rollout/use/schema_param_reward", _collect_component(use_states, "schema_param_reward"))
            _append_mean("rollout/use/tool_exec_raw", _collect_component(use_states, "tool_exec_raw"))
            _append_mean("rollout/use/tool_exec_norm", _collect_component(use_states, "tool_exec_norm"))
            _append_mean("rollout/use/tool_exec_scaled", _collect_component(use_states, "tool_exec_scaled"))
            _append_mean("rollout/use/efficiency_multiplier", _collect_component(use_states, "efficiency_multiplier"))
            _append_mean("rollout/use/final_answer_reward", _collect_component(use_states, "final_answer_reward"))
            _append_mean("rollout/use/final_answer_correct_rate", _collect_component(use_states, "final_answer_correct"))
            _append_mean("rollout/use/no_answer_penalty", _collect_component(use_states, "no_answer_penalty"))
            _append_rate(
                "rollout/use/step1_failed_rate",
                sum(1 for st in use_states if st.get("step1_failure_reason")),
                len(use_states),
            )
            _append_rate(
                "rollout/use/step1_fail_missing_python_or_schema_rate",
                sum(1 for st in use_states if st.get("step1_failure_reason") == "missing_python_or_schema"),
                len(use_states),
            )
            _append_rate(
                "rollout/use/step1_fail_schema_missing_function_name_rate",
                sum(1 for st in use_states if st.get("step1_failure_reason") == "schema_missing_function_name"),
                len(use_states),
            )
            _append_rate(
                "rollout/use/step1_fail_schema_function_mismatch_rate",
                sum(1 for st in use_states if st.get("step1_failure_reason") == "schema_function_mismatch"),
                len(use_states),
            )
            _append_rate(
                "rollout/use/step1_fail_schema_param_mismatch_rate",
                sum(1 for st in use_states if st.get("step1_failure_reason") == "schema_param_mismatch"),
                len(use_states),
            )
            _pool_covered = sum(1 for st in use_states if st.get("pool_used"))
            _append_rate("rollout/use/pool_coverage_rate", _pool_covered, len(use_states))

        _append_mean("rollout/pool/total_tools", [float(tool_pool.total_tools())])

        # Per-task reward breakdown (Q3: task difficulty correlation)
        task_reward_map: dict[str, list[float]] = {}
        for i, state in enumerate(states):
            task_cat = _task_category_from_meta(metas[i] if i < len(metas) else {}) or "unknown"
            if task_cat not in task_reward_map:
                task_reward_map[task_cat] = []
            task_reward_map[task_cat].append(float(state.get("env_reward") or 0.0))
        for task_cat, rewards in task_reward_map.items():
            safe_name = task_cat.replace("-", "_")
            _append_mean(f"rollout/by_task/{safe_name}/env_reward", rewards)

    if log_event is not None:
        log_event(
            "rollout_batch_end",
            rollout_batch_id=rollout_batch_id,
            num_prompts=len(states),
            num_turns_mean=(sum(num_turns_list) / len(num_turns_list)) if num_turns_list else 0.0,
            num_turns_min=min(num_turns_list) if num_turns_list else 0,
            num_turns_max=max(num_turns_list) if num_turns_list else 0,
            judge_reward_mean=(sum(judge_reward_list) / len(judge_reward_list)) if judge_reward_list else 0.0,
        )

    return {
        "prompt_ids": prompt_ids_list,
        "completion_ids": completion_ids_list,
        "logprobs": logprobs_list,
        "env_mask": env_mask_list,
        "env_reward": env_reward_list,
        "use_correct_reward": use_correct_reward_list,
        "num_turns": num_turns_list,
        "task_type": task_type_list,
        "judge_reward": judge_reward_list,
    }
