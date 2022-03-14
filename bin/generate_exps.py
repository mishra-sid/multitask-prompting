import os 
import argparse
import wandb
import yaml

log_dir = 'logs'
slurm_path = 'slurm_logs'

def dict_to_key_val(d):
    return "_".join(f"{k}={v}"for k, v in d.items())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('sweep_config_file', type='str')

    parser.add_argument('project_name', type='str')
    
    args = parser.parse_args()
    with open(args.sweep_config_file) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(config_dict, project=args.project_name)
    
    unique_id = dict_to_key_val(config_dict.params)
    common = f"""#!/bin/bash
#SBATCH --job-name=softprompting-{unique_id}
#SBATCH -o {slurm_path}/logs_{unique_id}.out
#SBATCH -e {slurm_path}/logs_{unique_id}.err
#SBATCH --time=0-4:00:00
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --exclude=node030"""

    with open(os.path.join(log_dir, f'script_{unique_id}.sh'), 'w') as f:
        f.write(common + "\n")
        f.write(f"wandb agent --count 1 ${sweep_id}\n")

