# Multi task prompting

## Installation
To install, in the root directory do:

```
conda env create -f environment.yml
conda activate multitask-prompting-env
pip install -e .
```

## Usage 
The training file can be used using:
```
multitask-prompting --help
usage: multitask-prompting [-h] [--output_dir OUTPUT_DIR] [--wandb WANDB] [--project_name PROJECT_NAME] [--cuda CUDA] [--seed SEED] [--task TASK] [--dataset DATASET] [--model MODEL] [--base_plm_family BASE_PLM_FAMILY] [--base_plm_path BASE_PLM_PATH] [--tune_plm TUNE_PLM] [--test_split TEST_SPLIT]
                           [--valid_split VALID_SPLIT] [--prompt_text PROMPT_TEXT] [--num_classes NUM_CLASSES] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--eval_every EVAL_EVERY] [--optimizer OPTIMIZER] [--warmup WARMUP]
                           [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--max_steps MAX_STEPS] [--weight_decay WEIGHT_DECAY] [--do_train DO_TRAIN] [--do_eval DO_EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
  --wandb WANDB
  --project_name PROJECT_NAME
  --cuda CUDA
  --seed SEED
  --task TASK
  --dataset DATASET
  --model MODEL
  --base_plm_family BASE_PLM_FAMILY
  --base_plm_path BASE_PLM_PATH
  --tune_plm TUNE_PLM
  --test_split TEST_SPLIT
  --valid_split VALID_SPLIT
  --prompt_text PROMPT_TEXT
  --num_classes NUM_CLASSES
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --eval_every EVAL_EVERY
  --optimizer OPTIMIZER
  --warmup WARMUP
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
  --max_steps MAX_STEPS
  --weight_decay WEIGHT_DECAY
  --do_train DO_TRAIN
  --do_eval DO_EVAL
```
