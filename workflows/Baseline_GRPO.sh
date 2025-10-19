#!/bin/bash
#SBATCH --job-name=MediPhi-GRPO
#SBATCH --partition=gpu
#SBATCH --constraint="a100|l40s|a40|rtx_8000"   # any available GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/MediPhi-GRPO.out

# -------------------------------
# ENVIRONMENT SETUP
# -------------------------------
module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs (before export): $CUDA_VISIBLE_DEVICES"
echo "Current working dir: $(pwd)"
echo "=================="

# -------------------------------
# FIX GPU VISIBILITY
# -------------------------------
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
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
print("=============================\n")
EOF

# -------------------------------
# RUN TRAINING-FREE GRPO EVALUATION
# -------------------------------
python workflows/baseline_grpo_single_agent.py \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/MediPhi-Instruct \
  --test_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
  --results_dir outputs/grpo_eval/MediPhi-Instruct_eval \
  --num_candidates 5 \
  --memory_size 3 \
  --unknown_reward 0.2 \
  --temperature 0.7 \
  --top_p 0.9 \
  --mode grpo
