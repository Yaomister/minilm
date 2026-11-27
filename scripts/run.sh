#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/llm_training_%j.out

cd $SLURM_SUBMIT_DIR

source /home/yao.eric/llm/.venv/bin/activate

python -u -m training.pretraining --depth 12
