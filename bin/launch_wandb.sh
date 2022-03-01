#!/bin/bash
#SBATCH --job-name=softprompting-{ind}
#SBATCH -o {slurm_path}/logs_{ind}.out
#SBATCH -e {slurm_path}/logs_{ind}.err
#SBATCH --time=0-4:00:00
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --exclude=node030


./bin/launch_agents.sh $SWEEP_ID 

