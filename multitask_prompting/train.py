import argparse
from multitask_prompting.trainer import Trainer

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm

from multitask_prompting.model import get_model
from multitask_prompting.data_utils import load_dataset, get_tokenized_dataloader, load_datasets_scenario
import torch
from multitask_prompting import utils
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_dir', type=str, default='models')
    
    # experiment config
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default='multitask_prompting')
    parser.add_argument('--device', type=str, default='cuda:0')
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
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--valid_split", type=float, default=0.3)
    parser.add_argument('--prompt_text', type=str, default='{"soft": None, "duplicate": 20}{"placeholder":"text_a"}{"mask"}.')
    parser.add_argument('--verbalizer_init', type=str, default='random', choices=['random', 'raw', 'first', 'last'])
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--verbalizer_input", type=str, default= 'mask',help='mask for usual warp, avg for avg of tokens instead')
    parser.add_argument("--train_per_scenario", type=bool, default= False,help='whether to train for all scenarios separately')

    
    # hyperparams
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--prompt_learning_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)

    # optimizer 
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # steps
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()
    print(args.__dict__)

    set_seed(args.seed)

    plm, tokenizer, model_config, wrapper_class = load_plm(args.base_plm_family, args.base_plm_path)

    if args.train_per_scenario ==False:
        print("Not training per scenario")
        metadata, train_raw_dataset, eval_raw_dataset, test_raw_dataset = load_dataset(args)
        model = get_model(args.task, args.model)(args, plm, metadata, tokenizer, model_config, wrapper_class)
        train_dataloader, valid_dataloader, test_dataloader = get_tokenized_dataloader(args, train_raw_dataset, eval_raw_dataset, test_raw_dataset, model.tokenizer, model.template, model.wrapper_class)

        trainer = Trainer(args, model)
        if args.do_train:
            trainer.train(train_dataloader, valid_dataloader, test_dataloader)
    
    else:

        
        metadata_, train_raw_dataset, eval_raw_dataset, test_raw_dataset = load_dataset(args)
        global_model = get_model(args.task, args.model)(args, plm, metadata_, tokenizer, model_config, wrapper_class)
        train_dataloader, valid_dataloader, test_dataloader = get_tokenized_dataloader(args, train_raw_dataset, eval_raw_dataset, test_raw_dataset, global_model.tokenizer, global_model.template, global_model.wrapper_class)
        trainer = Trainer(args, global_model)
        if args.do_train:
            # trainer.train(train_dataloader, valid_dataloader, test_dataloader)
            path = Path(args.model_dir) / utils.get_uniq_str(args) / "model.pth" 
            global_model.load_state_dict(torch.load(path), strict=False)

        
        metadata, raw_datasets = load_datasets_scenario(args)
        for scenario in metadata.keys():
            print("Training model for scenario: ", scenario)
            metadata_scenario, train_raw_dataset, eval_raw_dataset, test_raw_dataset = metadata[scenario], raw_datasets[scenario]['train'], raw_datasets[scenario]['valid'], raw_datasets[scenario]['test']
            model = get_model(args.task, args.model)(args, plm, metadata_scenario, tokenizer, model_config, wrapper_class,global_model)
            train_dataloader, valid_dataloader, test_dataloader = get_tokenized_dataloader(args, train_raw_dataset, eval_raw_dataset, test_raw_dataset, model.tokenizer, model.template, model.wrapper_class)

            trainer = Trainer(args, model)
            if args.do_train:
                trainer.train(train_dataloader, valid_dataloader, test_dataloader,scenario)



