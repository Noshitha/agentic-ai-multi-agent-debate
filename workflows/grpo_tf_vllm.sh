#!/bin/bash
#SBATCH --job-name=grpo_vllm_4gpu
#SBATCH --partition=gpu
#SBATCH --constraint="v100|a100|l40s"
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --qos=long
#SBATCH --output=logs/grpo_vllm_4gpu.out

module load cuda/12.1
source ~/venvs/torch_env/bin/activate

if ! python -c "import vllm" 2>/dev/null; then
  echo "Installing vLLM..."
  pip install --no-input vllm
else
  echo "vLLM already installed."
fi

export VLLM_WORKER_MULTIPROCESSING_METHOD=spawn
export VLLM_TP=4

echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Current working dir: $(pwd)"
echo "=================="

echo "==== ENV INFO ===="
python -V
pip show torch | grep -E 'Name|Version'
pip show transformers | grep -E 'Name|Version'
nvidia-smi
echo "=================="

cd /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows

START_TS=$(date +%s)

python -m grpo_tf.train \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B \
  --dataset /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/sdoh-mad-baselines/workflows/dataset/alcohol/test_debug.jsonl \
  --experiment_name alcohol_grpo \
  --epochs 1 \
  --batchsize 10 \
  --grpo_n 2 \
  --temperature 0.7 \
  --max_tokens 256 \
  --rollout_concurrency 2

END_TS=$(date +%s)
echo "TOTAL_SECONDS=$((END_TS - START_TS))"
