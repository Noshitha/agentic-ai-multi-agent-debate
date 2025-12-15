#!/bin/bash
#SBATCH --job-name=grpo_vllm_1gpu
#SBATCH --partition=gpu
#SBATCH --constraint="v100|a100|l40s"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grpo_vllm_1gpu_debug.out

module load cuda/12.1
source ~/venvs/torch_env/bin/activate

export VLLM_WORKER_MULTIPROCESSING_METHOD=spawn
export VLLM_TP=1

cd /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows

START_TS=$(date +%s)

python -m grpo_tf.train \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B \
  --dataset /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/dataset/alcohol/test_debug.jsonl \
  --experiment_name alcohol_grpo_vllm_1gpu_debug \
  --epochs 1 \
  --batchsize 4 \
  --grpo_n 2 \
  --temperature 0.7 \
  --max_tokens 256 \
  --rollout_concurrency 2

END_TS=$(date +%s)
echo "TOTAL_SECONDS=$((END_TS - START_TS))"
