#!/bin/bash

#SBATCH --job-name=vanilla
#SBATCH -p titanxp
#SBATCH -N 2
#SBATCH  --nodelist=n2
#SBATCH --gres=gpu:2

# epoch is enough around 200~300

srun python main.py  --epochs=200
