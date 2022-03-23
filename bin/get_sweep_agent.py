import wandb
import argparse
import yaml
import subprocess

with open("../multitask_prompting/config/config_trial.yaml") as file:
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = '../multitask_prompting/train.py'

sweep_id = wandb.init(config_dict, project= 'trial_sweeps')
print(sweep_id)

