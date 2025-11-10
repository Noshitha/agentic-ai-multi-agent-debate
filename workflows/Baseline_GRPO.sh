#!/bin/bash
#SBATCH --job-name=MediPhi-GRPO
#SBATCH --partition=gpu
#SBATCH --constraint="a100|l40s|a40|rtx_8000|v100"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=08:00:00
#SBATCH --output=logs/MediPhi-GRPO.out

module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PWD: $(pwd)"
echo "=================="

echo "==== ENV INFO ===="
python -V
pip show torch transformers | grep -E 'Name|Version'
nvidia-smi
echo "=================="

python - <<'EOF'
import torch
torch.cuda.empty_cache()
print("\n==== GPU Cache Cleared ====\n")
EOF

python - <<'EOF'
import torch
print("\n==== PYTORCH CUDA CHECK ====")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
print("=============================\n")
EOF

python workflows/Baseline_GRPO_Single_Agent.py \
  --model_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/MediPhi-Instruct \
  --test_path /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/dataset/alcohol/test.jsonl \
  --results_dir outputs/rag_grpo_eval/MediPhi-Instruct_eval \
  --num_candidates 2 \
  --unknown_reward 0.2 \
  --temperature 0.7 \
  --top_p 0.9 \
  --k_retrieve 3

python - <<'EOF'
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print("\n==== Job complete: GPU memory cleaned ====\n")
EOF
