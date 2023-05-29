#!/bin/bash

#SBATCH --job-name=cutmix
#SBATCH -p cpu-max10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q nogpu
#SBATCH --cpus-per-task=10

# epoch is enough around 200~300

srun python main.py  --function=attack --aug_type=cutmix
