# Supplement

This directory contains the code and data referenced by the paper. It is organized into three sub-folders.

```
supplement/
├── data/
│   └── tabmwp_hard.zip          # TabMWP-Hard dataset (5 k examples)
├── tasks/                        # 23-task algorithmic dataset creation
│   ├── create_all_datasets.py
│   ├── utils.py
│   ├── __init__.py
│   └── <23 task files>.py
└── tool_rl/                      # Core RL training code
    ├── decompose_rollout_manual_refact_w_judge_rewards_with_tool_pool_lora_sync.py
    ├── judge_utils.py
    ├── rewards.py
    ├── tools.py
    ├── decompose_rollout_utils.py
    ├── decompose_single_q_rollout_external.py
    ├── llm_judge_eq.py
    ├── xgrammar_schema_check.py
    └── run_tool_rl_decompose_4b_judge_external_lora_sync.sh
```

---

## data/ — TabMWP-Hard dataset

**Paper section:** Appendix — TabMWP-Hard construction.

`tabmwp_hard.zip` contains `tabmwp_hard_hf_5k_fixed.jsonl` (5,000 examples).

To unzip:

```bash
cd supplement/data
unzip tabmwp_hard.zip
```

Each line is a JSON object with the fields `question`, `answer`, `table`, and metadata used
for the OOD inference evaluation described in the paper.

---

## tasks/ — 23 algorithmic task dataset creation

**Paper section:** OOD inference protocol / algorithmic task suite.

The 23 tasks are:
`ab`, `base_conversion`, `bitwise_arithematic`, `caesar_cipher`, `calendar_arithmetic`,
`chain_sum`, `count_bits`, `countdown`, `cryptarithm`, `gcd`, `isomorphic_string`,
`knights_knaves`, `lcm`, `polynomial_equations`, `polynomial_multiplication`,
`prime_factorization`, `propositional_logic`, `simple_equations`, `simple_integration`,
`spell_backward`, `syllogism`, `tower_of_hanoi`, `word_ladder`.

Each task file generates train / validation / test splits using
[reasoning_gym](https://github.com/open-thought/reasoning-gym) as the only non-stdlib
dependency.

**To regenerate all 23 task datasets locally (first time only):**

```bash
pip install reasoning_gym
python supplement/tasks/create_all_datasets.py
```

> **Note:** Dataset creation only needs to be run once. Each task's `init_data()` generates
> and pushes the train / validation / test splits to HuggingFace Hub. Re-running will
> overwrite the existing splits. After the first run, load the datasets directly via
> `load_dataset(...)` (see `load_ds()` in each task file) without re-generating them.

Each split is deduplicated by question text across difficulty levels before being pushed,
so overlapping questions generated at different difficulty settings are collapsed to their
first (easiest) occurrence.

---

## tool_rl/ — Core RL training code

**Paper section:** Tool-RL training (Section 3 / Appendix).

The entry point is the rollout module:

```
decompose_rollout_manual_refact_w_judge_rewards_with_tool_pool_lora_sync.py
```

It implements the two-phase (build + use) GRPO rollout with:
- a **LoRA-synced evaluator** (port 9002) that scores tool usability and is updated after
  each checkpoint,
- a **fixed judge** (port 9003) that scores code quality/clarity throughout training.

### Dependencies

```
pip install trl vllm openai xgrammar reasoning_gym
```

### Minimal file dependency graph

```
decompose_rollout_manual_refact_w_judge_rewards_with_tool_pool_lora_sync.py
├── rewards.py          ← answer-matching + llm_judge_eq
├── judge_utils.py      ← judge call helpers
├── xgrammar_schema_check.py
├── decompose_single_q_rollout_external.py
│   └── decompose_rollout_utils.py
│       └── tools.py   ← sandboxed code execution
└── llm_judge_eq.py    ← LLM-as-judge equality check
```

### Running training

The shell script `run_tool_rl_decompose_4b_judge_external_lora_sync.sh` reproduces the
main experiment from the paper (Qwen3-4B, LoRA r=64, DAPO, 60 steps).

Before running, start the two vLLM servers:

```bash
# Evaluator (LoRA-sync, port 9002) — must have LORA runtime updating enabled
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 --port 9002 --enable-lora

# Fixed judge (port 9003)
python -m vllm.entrypoints.openai.api_server \
    --model NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 --port 9003
```

Then launch training:

```bash
bash supplement/tool_rl/run_tool_rl_decompose_4b_judge_external_lora_sync.sh
```

Set `--report_to wandb` and supply `--wandb_entity <your_entity>` if you want W&B logging,
or remove those flags to disable it.

The code-execution sandbox (`tools.py`) defaults to an in-process subprocess runner.
Set `TOOL_RL_USE_SANDBOX=1` and `TOOL_RL_SANDBOX_URL` to point at a remote sandboxed
executor if you want isolated execution.
