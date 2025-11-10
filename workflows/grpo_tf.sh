#!/bin/bash
#SBATCH --job-name=grpo_tf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grpo_tf.out

# Load CUDA if needed
module load cuda/12.1

# Activate your environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# -------------------------------
# Print debug info to log
# -------------------------------
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
# -------------------------------
# Run GRPO-TF 
# -------------------------------
python -m grpo_tf.train \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B \
  --dataset /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
  --experiment_name alcohol_grpo \
  --epochs 3 \
  --batchsize 10 \
  --grpo_n 5 \
  --temperature 0.7 \
  --max_tokens 256
