from pathlib import Path 

log_dir = 'bin_out'
slurm_path = 'slurm_logs'
conda_path = "/home/siddharthami/anaconda3/etc/profile.d/conda.sh"

def arg_to_key_val(arg):
    return "_".join(f"{a.split(' ')[0]}={' '.join(a.split(' ')[1:]).replace(' ', '_')}" for a in arg)

args = [
    ['--base_plm_family bert', '--base_plm_path bert-base-cased', '--verbalizer_init random', '--prompt_text {"soft": None, "duplicate": 1}{"placeholder":"text_a"}{"mask"}'],
    ['--base_plm_family bert', '--base_plm_path bert-base-cased', '--verbalizer_init random', '--prompt_text {"soft": None, "duplicate": 20}{"placeholder":"text_a"}{"mask"}'],
    ['--base_plm_family roberta', '--base_plm_path roberta-large', '--verbalizer_init random', '--prompt_text {"soft": None, "duplicate": 1}{"placeholder":"text_a"}{"mask"}'],
    ['--base_plm_family roberta', '--base_plm_path roberta-large', '--verbalizer_init random', '--prompt_text {"soft": None, "duplicate": 20}{"placeholder":"text_a"}{"mask"}'],
]

if __name__ == "__main__":
    for arg in args:
        unique_id = arg_to_key_val(arg)
        common = f"""#!/bin/bash
#SBATCH --job-name=multitask-prompting-{unique_id}
#SBATCH -o {slurm_path}/logs_{unique_id}.out
#SBATCH -e {slurm_path}/logs_{unique_id}.err
#SBATCH --time=0-4:00:00
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

. {conda_path}
conda activate multitask-prompting
"""
        out_path = Path(log_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f'script_{unique_id}.sh', 'w') as f:
            f.write(common + "\n")
            f.write(f"multitask-prompting {' '.join(arg)}\n")

