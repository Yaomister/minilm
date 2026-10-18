#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%A_%a.out

cd $SLURM_SUBMIT_DIR

mkdir -p logs

source /home/yao.eric/llm/.venv/bin/activate

python -u -m training.pretraining
