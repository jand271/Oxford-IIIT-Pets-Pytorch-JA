#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --cpus-per-task=1
#SBATCH --job-name=try_spoof_resnet
#SBATCH --gres=gpu:1

module purge
module load python/3.6.1
module load cuda/11.0 

source venv/bin/activate

python try_spoof_resnet.py
