#!/usr/bin/env bash
# Uses --rollout lora_sync WITH --judge_sync_lora, so the evaluator model
# (localhost:9002) has its LoRA adapter hot-swapped after each checkpoint save.
# This gives build task scoring a progressively improving evaluator that tracks
# the training model's current quality.
# The judge model (port 9003) remains fixed throughout for code quality scoring.
# The evaluator vLLM server must be started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# and --enable-lora (see scripts/serve_judge_vllm.sh).
# Contrast: run_tool_rl_decompose_4b_judge_external.sh uses a fixed evaluator (no LoRA sync).

set -euo pipefail

export TOOL_RL_USE_SANDBOX=1
export TOOL_RL_SANDBOX_URL="http://localhost:8080/run_code"
MODEL="Qwen/Qwen3-4B-Instruct-2507"

CUDA_VISIBLE_DEVICES=0 \
  python -m tool_rl.train_decompose_grpo_alternating \
    --model "$MODEL" \
    --dataset hf \
    --hf_dataset <anonymous>/reasoning-gym-merged-with-samples-3 \
    --output_dir outputs/2-task_decompose_qwen3-4b_lora-r64-lora-sync-5-pool_dapo_lr6e-5-max-step-60-sample8-bs48-agd \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_generations 8 \
    --temperature 0.7 \
    --top_p 0.95 \
    --min_p 0.01 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 128 \
    --max_completion_length 4096 \
    --max_prompt_length 5120 \
    --save_steps 5 \
    --num_few_shot 5 \
    --beta 0.01 \
    --num_test_questions 16 \
    --warmup_steps 10 \
    --max_steps 60 \
    --logging_steps 1 \
    --build_fraction 0.5 \
    --phase2_max_completion_ratio 0.2 \
    --rollout lora_sync \
    --tool_pool_max_per_task 20 \
    --tool_pool_max_provide 3 \
    --save_total_limit 30 \
    --tool_pool_distractor_count 2 \
    --use_judge_reward \
    --judge_base_url "http://localhost:9003/v1" \
    --judge_model "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4" \
    --judge_api_key "TEST_API" \
    --judge_reward_weight 0.2 \
    --judge_max_workers 64 \
    --judge_timeout_s 30 \
    --judge_max_questions 3 \
    --evaluator_base_url "http://localhost:9002/v1" \
    --evaluator_backend sglang \
    --evaluator_model "Qwen/Qwen3-4B-Instruct-2507" \
    --evaluator_api_key "TEST_API" \
    --judge_sync_lora \
    --vllm_mode colocate \
    --vllm_max_model_length 11240 \
    --vllm_gpu_memory_utilization 0.2 \
    --report_to wandb \
    --wandb_project TOOL-RL \
    --wandb_entity <anonymous> \
    --wandb_run_name "2-task_decompose_qwen3-4b_lora-r64-lora-sync-5-pool_dapo_lr6e-5-max-step-60-sample8-bs48-agd" \
    --num_completions_to_print 0 \
    --log_rollout_messages \
    --learning_rate 6e-5
