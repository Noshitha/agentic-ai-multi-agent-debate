#!/bin/bash
#SBATCH --job-name=qwen_agentic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_agentic_%j.out

module load cuda/12.1
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# -------------------------------
# Print debug info to log
# -------------------------------
echo "==== JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model Path: /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B"
echo "Current working dir: $(pwd)"
echo "=================="

echo "==== ENV INFO ===="
python -V
pip show torch | grep -E 'Name|Version'
pip show transformers | grep -E 'Name|Version'
nvidia-smi
echo "=================="

python workflows/qwen_agentic_baseline.py \
  --model_id /project/pi_hongyu_umass_edu/zonghai/sdoh_agentic/models/Qwen3-0.6B \
  --prompt_file prompts/alcohol_prompt.txt \
  --results_dir outputs/qwen3_0.6B \
  --num_agents 3
