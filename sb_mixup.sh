#!/bin/bash

#SBATCH --job-name=mixup
#SBATCH -p titanxp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2

# epoch is enough around 200~300

srun python main.py --epochs=200 --batch_size=128 --aug_type=mixup
