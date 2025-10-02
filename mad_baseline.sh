#!/bin/bash
#SBATCH --job-name=mad_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mad_baseline_%j.out

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

# -------------------------------
# Run MAD baseline
# -------------------------------
python mad_baseline.py \
  --model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --prompt_file ../alcohol/prompt_for_gpt/prompt_initial_step.txt \
  --results_dir results/mad_baseline/alcohol \
  --num_agents 3

