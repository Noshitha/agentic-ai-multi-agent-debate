#!/bin/bash
#SBATCH --job-name=MediPhi-Instruct
#SBATCH --partition=gpu
#SBATCH --constraint="a100|l40s|a40|rtx_8000"     # pick any of these free GPU types
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/MediPhi-Instruct.out

# -------------------------------
# ENVIRONMENT SETUP
# -------------------------------
module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs (before export): $CUDA_VISIBLE_DEVICES"
echo "Model Path: /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/MediPhi-Instruct"
echo "Current working dir: $(pwd)"
echo "=================="

# -------------------------------
# FIX GPU VISIBILITY
# -------------------------------
# Ensure CUDA is visible to PyTorch
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES (after export): $CUDA_VISIBLE_DEVICES"

# -------------------------------
# PYTHON ENVIRONMENT CHECKS
# -------------------------------
echo "==== ENV INFO ===="
python -V
pip show torch | grep -E 'Name|Version'
pip show transformers | grep -E 'Name|Version'
nvidia-smi
echo "=================="

# -------------------------------
# QUICK GPU SANITY CHECK
# -------------------------------
python - <<'EOF'
import torch
print("\n==== PYTORCH CUDA CHECK ====")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
print("=============================\n")
EOF

# -------------------------------
# RUN BASELINE SCRIPT
# -------------------------------
python workflows/baseline_qwen_single_Agent.py \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/MediPhi-Instruct \
  --train_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/train.jsonl \
  --test_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
  --results_dir outputs/MediPhi-Instruct-evals