#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# slurm opens this file before the script runs, so logs/ has to already exist (it's kept in the repo)
#SBATCH --output=logs/train_%j.out

cd $SLURM_SUBMIT_DIR

source /home/yao.eric/llm/.venv/bin/activate

python -u -m training.pretraining --depth 12
