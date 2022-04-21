import argparse
from multitask_prompting.trainer import Trainer

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm

from multitask_prompting.model import get_model
from multitask_prompting.data_utils import load_nlu_dataset, get_tokenized_dataloader
from multitask_prompting.data_utils_super_glue import load_super_glue_dataset

def main():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_dir', type=str, default='models')
    
    # experiment config
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default='multitask_prompting')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=42)
    
    # args
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--dataset', type=str, default='nlu_evaluation_data')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='warp')
    parser.add_argument('--model_type', type=str, default='prompt')
    parser.add_argument('--base_plm_family', type=str, default='bert')
    parser.add_argument('--base_plm_path', type=str, default='bert-base-uncased')
    parser.add_argument("--tune_plm", type=bool, default=False)
    parser.add_argument("--share_plm_weights", type=bool, default=True)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--valid_split", type=float, default=0.3)
    parser.add_argument('--prompt_num_soft_tokens', type=int, default=20)
    parser.add_argument('--verbalizer_init', type=str, default='random', choices=['random', 'raw', 'first', 'last'])
    parser.add_argument("--max_seq_length", type=int, default=64)
    
    # hyperparams
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--prompt_learning_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)

    # optimizer 
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # steps
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    args = parser.parse_args()
    print(args.__dict__)

    set_seed(args.seed)

    plm, tokenizer, model_config, wrapper_class = load_plm(args.base_plm_family, args.base_plm_path)
    if args.dataset == "super_glue":
        metadata, raw_datasets = load_super_glue_dataset(args)
    else:
        metadata, raw_datasets = load_nlu_dataset(args)

    model = get_model(args.task, args.model)(args, plm, metadata, tokenizer, model_config, wrapper_class)
    dataloaders = get_tokenized_dataloader(args, metadata, raw_datasets, model.tokenizer, model.templates, model.wrapper_class)

    trainer = Trainer(args, metadata, model)
    if args.do_train:
        trainer.train(dataloaders)
    
    if args.do_test:
        trainer.test(args, model, dataloaders)
