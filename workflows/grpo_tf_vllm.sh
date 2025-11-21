#!/bin/bash
#SBATCH --job-name=grpo_tf_vLLM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/grpo_tf_vLLM.out

module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

export VLLM_WORKER_MULTIPROCESSING_METHOD=spawn

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

python -m grpo_tf.train \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B \
  --dataset /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
  --experiment_name alcohol_grpo \
  --epochs 1 \
  --batchsize 10 \
  --grpo_n 5 \
  --temperature 0.7 \
  --max_tokens 256 \
  --rollout_concurrency 2
