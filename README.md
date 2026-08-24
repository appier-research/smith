# SMITH: Joint Optimization of Tool Creation and Use for Large Language Model Agents

**Schema-grounded Multi-task Iterative Tool Honing**

[Zhi Rui Tam](https://zrt.wtf)<sup>1,2</sup>, Chieh-Yen Lin<sup>1</sup>, Yun-Nung (Vivian) Chen<sup>2</sup>, Shao-Hua Sun<sup>1,2</sup>, Hung-yi Lee<sup>2</sup>

<sup>1</sup>Appier AI Research &nbsp;&nbsp; <sup>2</sup>National Taiwan University

[[Paper]](https://tool-use-smith.github.io) [[Project Page]](https://tool-use-smith.github.io) [[Code]](https://github.com/appier-research/smith)

---

![SMITH teaser: the same policy invents a tool, uses it to solve problems, and learns from the results.](https://raw.githubusercontent.com/tool-use-smith/tool-use-smith.github.io/main/static/teaser.png)

SMITH is a training loop, not a fixed pipeline: the *same* policy invents a tool, uses it to solve problems, and is optimized on whether that use succeeds. That feedback is what keeps tool creation improving over time.

---

## Abstract

Tool-augmented language models are bounded by the APIs humans bothered to write; existing tool-creation systems patch this by prompting a frozen LLM at inference time, leaving the model that writes a tool decoupled from the one that uses it, with no signal that the schemas it produces are schemas it can actually invoke. We propose **SMITH** (Schema-grounded Multi-task Iterative Tool Honing), a reinforcement learning framework that jointly trains tool creation and tool use inside a single policy. Each rollout is either a *build* task (write a tool from a few examples) or a *use* task (invoke a pooled tool on a held-out question). Three separate reward axes catch schema, code, and outcome failures independently, so each failure mode contributes its own gradient. A 4B Qwen3 trained with SMITH on 13 procedural reasoning tasks with exact verifiers reaches **79.9** macro-average accuracy on held-out tasks, the best across all evaluated methods and ahead of an untrained 30B-A3B tool-writer. It also reaches **40.4** on TabMWP-Hard and **42.6** on out-of-domain GQA (**+7.6** over the best same-backbone inference-time baseline), without any visual or tabular training data. When invoked by a frozen 350M student, tools written by our 4B match those produced by a writer an order of magnitude larger.

**Key results:**
- **79.9** macro-average accuracy on held-out Reasoning-Gym tasks (best of all methods)
- **32×** fewer output tokens than standard CoT
- **+7.6** points on out-of-domain GQA vs. best same-backbone baseline
- **42.9** held-out accuracy for a 350M model using SMITH's tools

---

## Method

![SMITH method diagram: build tasks produce a Python tool and JSON schema, tools are validated and stored in a pool, and use tasks later invoke a pooled tool on a held-out question, feeding execution-based reward back to the policy.](https://raw.githubusercontent.com/tool-use-smith/tool-use-smith.github.io/main/static/img/method-diagram.png)

SMITH is a multi-task RL framework trained with DAPO (a clip-higher variant of GRPO) that mixes two rollout types into every batch: **build** and **use**. Both are optimized inside the *same* policy, so gradients from tool creation and tool consumption update the same weights every step.

### Build task

The policy sees **N=4** question–answer pairs and must infer the general procedure behind them, then express it as an OpenAI-compatible `(Python function, JSON schema)` pair. The tool is run against **K=16** held-out questions drawn from a harder difficulty band; tools that pass are stored in a shared pool.

### Use task

The policy receives a target question and a pool entry — one matching domain tool plus two distractor tools — and has up to **T=5** turns to call the tool and answer. An efficiency penalty discourages burning through the turn budget.

### Three reward axes

| Axis | What it checks |
|---|---|
| **Format** | Response contains exactly one Python block and one JSON block whose function name and parameters match |
| **Evaluation** | A LoRA-synced evaluator model must call the tool to answer each held-out question; only tool-call answers count |
| **Judge** | An LLM judge scores code correctness, schema quality, and overall quality as a separate axis |

The evaluator and judge both start from the same base checkpoint as the policy, then are periodically re-synced to the latest policy weights, breaking the instability of scoring against live weights.

---

## Results

### Reasoning-Gym (held-out tasks)

| Method | Seen Avg | Unseen Avg | I/O tokens |
|---|---|---|---|
| Standard CoT | 58.0 | 55.7 | 173 / 3,206 |
| LATM (Qwen3-4B) | 77.6 | 58.3 | 607 / 174 |
| LATM (Qwen3-30B-A3B) | 74.0 | 74.1 | 659 / 405 |
| CRAFT | 74.1 | 76.5 | 1,226 / 418 |
| TroVE | 52.6 | 55.9 | 347 / 575 |
| KTCE | 61.0 | 65.1 | 319 / 404 |
| ReTool (distill Qwen-32B) | **92.2** | 63.2 | 1,707 / 633 |
| LATM (distill GPT-4.1) | 81.7 | 65.8 | 638 / 207 |
| **SMITH (ours)** | 85.2 | **79.9** | 664 / **100** |

### Out-of-domain generalization

| Method | TabMWP-Hard | GQA |
|---|---|---|
| Standard CoT | 7.2 | 11.5 |
| LATM (Qwen3-4B) | 19.7 | 35.0 |
| CRAFT | 30.0 | 21.9 |
| TroVE | 36.4 | 21.4 |
| ReTool | 3.0 | 26.1 |
| LATM (distill GPT-4.1) | 7.1 | **56.0** |
| **SMITH (ours)** | **40.4** | 42.6 |

### Cross-backbone scaling

| Method | RG (Seen) | RG (Unseen) | TabMWP | GQA |
|---|---|---|---|---|
| Qwen3-8B (base) | 72.6 | 72.2 | 42.4 | 17.3 |
| **SMITH: Qwen3-8B** | **79.4** | **81.7** | **56.7** | **28.7** |
| Granite-3.3-8B (base) | 31.2 | 22.0 | 3.9 | 7.8 |
| **SMITH: Granite-3.3-8B** | 39.1 | 28.5 | 4.5 | 11.7 |

---

## Repository structure

```
smith/
├── data/
│   └── tabmwp_hard.zip          # TabMWP-Hard dataset (5,000 examples)
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

## Data

### TabMWP-Hard

`data/tabmwp_hard.zip` contains `tabmwp_hard_hf_5k_fixed.jsonl` (5,000 examples), a strengthened tabular-reasoning benchmark used for OOD evaluation.

```bash
cd data && unzip tabmwp_hard.zip
```

Each line is a JSON object with the fields `question`, `answer`, `table`, and metadata.

### Reasoning-Gym task datasets

The 23 training/evaluation tasks are:
`ab`, `base_conversion`, `bitwise_arithematic`, `caesar_cipher`, `calendar_arithmetic`,
`chain_sum`, `count_bits`, `countdown`, `cryptarithm`, `gcd`, `isomorphic_string`,
`knights_knaves`, `lcm`, `polynomial_equations`, `polynomial_multiplication`,
`prime_factorization`, `propositional_logic`, `simple_equations`, `simple_integration`,
`spell_backward`, `syllogism`, `tower_of_hanoi`, `word_ladder`.

Each task file generates train/validation/test splits using [reasoning_gym](https://github.com/open-thought/reasoning-gym).

```bash
pip install reasoning_gym
python tasks/create_all_datasets.py
```

> **Note:** Dataset creation only needs to be run once. Each task's `init_data()` pushes splits to HuggingFace Hub. After the first run, load datasets directly via `load_dataset(...)` (see `load_ds()` in each task file).

---

## Training

### Dependencies

```bash
pip install trl vllm openai xgrammar reasoning_gym
```

### File dependency graph

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

### Running

The shell script `run_tool_rl_decompose_4b_judge_external_lora_sync.sh` reproduces the main experiment (Qwen3-4B, LoRA r=64, DAPO, 60 steps).

First, start the two vLLM servers:

```bash
# Evaluator (LoRA-sync, port 9002)
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 --port 9002 --enable-lora

# Fixed judge (port 9003)
python -m vllm.entrypoints.openai.api_server \
    --model NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 --port 9003
```

Then launch training:

```bash
bash tool_rl/run_tool_rl_decompose_4b_judge_external_lora_sync.sh
```

Add `--report_to wandb --wandb_entity <your_entity>` for W&B logging.

The code-execution sandbox (`tools.py`) defaults to an in-process subprocess runner. Set `TOOL_RL_USE_SANDBOX=1` and `TOOL_RL_SANDBOX_URL` to point at a remote sandboxed executor for isolated execution.

---

## BibTeX

```bibtex
@article{tam2026smith,
  title   = {Joint Optimization of Tool Creation and Use for Large Language Model Agents},
  author  = {Tam, Zhi Rui and Lin, Chieh-Yen and Chen, Yun-Nung and Sun, Shao-Hua and Lee, Hung-yi},
  year    = {2026},
  journal = {arXiv preprint},
  url     = {https://tool-use-smith.github.io}
}
```
