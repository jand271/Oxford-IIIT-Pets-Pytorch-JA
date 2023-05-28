#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=try_spoof_resnet
#SBATCH -p gpu

module purge
module load python/3.6.1
module load cuda/11.0 

current_datetime=$(date +"%D %T")
echo "Current date and time: $current_datetime"
echo "Starting Script"
python try_spoof_resnet.py
echo "Ending Script"
echo "Current date and time: $current_datetime"
