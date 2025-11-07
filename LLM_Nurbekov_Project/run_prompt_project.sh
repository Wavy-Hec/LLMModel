#!/bin/bash
#SBATCH --job-name=prompt-injection
#SBATCH --output=prompt_logs.txt
#SBATCH --error=prompt_errors.txt
#SBATCH --partition=gpua30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

echo "---- ENVIRONMENT DEBUG ----"
echo "Job running on node: $(hostname)"
echo "Job started at: $(date)"
/home/alissenmoreno01/miniconda3/envs/promptenv/bin/python -V
/home/alissenmoreno01/miniconda3/envs/promptenv/bin/python -c "import sys, transformers; print('Python path:', sys.executable); print('Transformers version in job:', transformers.__version__)"
echo "----------------------------"

cd ~/LLM_Nurbekov_Project
/home/alissenmoreno01/miniconda3/envs/promptenv/bin/python project.py

echo "✅ Job finished at $(date)"
