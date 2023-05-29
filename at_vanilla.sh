#!/bin/bash

#SBATCH --job-name=vanilla
#SBATCH -p cpu-max24
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q nogpu
#SBATCH --cpus-per-task=10

# epoch is enough around 200~300

srun python main.py  --function=attack
