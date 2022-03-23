import argparse, os, json
parser = argparse.ArgumentParser(description='Run multiple wandb agents via SLURM')
parser.add_argument('--sweep_id', type=str)
parser.add_argument('--partition', type=str, default="gpu")
parser.add_argument('--exclude', type=str, default=None, help="e.g. node001,node002")
parser.add_argument('--njobs', type=int, default=1, help="number of agents to add to the sweep")
parser.add_argument('--nruns', type=int, default=10, help="number of runs per agent")
parser.add_argument('--wandb_account', type=str, default='softprompting')
parser.add_argument('--wandb_project', type=str, default='multitask-prompting-bin')
parser.add_argument('--slurm_dir', type=str, default="slurm", help="directory to store slurm logs")
parser.add_argument('--srun_filename', type=str, default="srun.sh", help="filename of srun script to create")
parser.add_argument('--sbatch_filename', type=str, default="sbatch.sh", help="filename of sbatch script to create")
args = parser.parse_args()
os.makedirs(args.slurm_dir, exist_ok=True)
with open(os.path.join(args.slurm_dir, '.gitignore'), 'w') as f:
    f.write('*\n!.gitignore')
sweep_dir = os.path.join(args.slurm_dir, args.sweep_id)
os.makedirs(sweep_dir, exist_ok=True)
sbatch_file_path = os.path.join(sweep_dir, args.sbatch_filename)
srun_file_path = os.path.join(sweep_dir, args.srun_filename)
with open(sbatch_file_path, 'w') as f:
    f.write(
        "\n".join((
            "#!/bin/bash",
            "#SBATCH --gres=gpu:1",
            f"#SBATCH --partition={args.partition}",
            f"#SBATCH --exclude={args.exclude}" if args.exclude else "",
            "#SBATCH --mem=20GB",
            f"#SBATCH --array=1-{args.njobs}" if args.njobs > 1 else "",
            f"#SBATCH --job-name={args.sweep_id}",
            f"#SBATCH --output={sweep_dir}/%A-%a.out" if args.njobs > 1 else f"#SBATCH --output={sweep_dir}/%A.out",
            f"srun {srun_file_path}"
        ))
    )
os.system(f"chmod +x {sbatch_file_path}")
with open(srun_file_path, 'w') as f:
    f.write(
        "\n".join((
            "#!/bin/sh",
            "module load cuda101/10.1.168 cudnn/7.6-cuda_10.1 gcc7/7.5.0",
            "export NO_PROGRESS_BAR=true",
            "hostname",
            f"wandb agent --count {args.nruns} {args.wandb_account}/{args.wandb_project}/{args.sweep_id}" \
                if args.nruns else f"wandb agent {args.wandb_account}/{args.wandb_project}/{args.sweep_id}",
        ))
    )
os.system(f"chmod +x {srun_file_path}")

os.system(f"sbatch {sbatch_file_path}")

