#!/bin/bash
#SBATCH --job-name=qwen_all_versions_a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen_a100_%j.out
# -------------------------------
# ENVIRONMENT SETUP
# -------------------------------
module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs (before export): $CUDA_VISIBLE_DEVICES"
echo "Model Path: /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B"
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

for MODEL in Qwen3_1.7B Qwen3_4B medgemma3-4b MediPhi-Instruct; do
  echo "Running evaluation for $MODEL..."
  python workflows/baseline_qwen_single_Agent.py \
    --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/$MODEL \
    --train_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/train.jsonl \
    --test_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
    --results_dir outputs/$MODEL\_eval
done