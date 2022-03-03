import argparse
from multitask_prompting.trainer import Trainer

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm

from multitask_prompting.model import get_model
from multitask_prompting.data_utils import load_dataset, get_tokenized_dataloader

def main():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--output_dir', type=str, default='output')
    
    # experiment config
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default='multitask_prompting')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    
    # args
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--dataset', type=str, default='nlu_evaluation_data')
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--base_plm_family', type=str, default='bert')
    parser.add_argument('--base_plm_path', type=str, default='bert-base-uncased')
    parser.add_argument("--tune_plm", type=bool, default=False)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--valid_split", type=float, default=0.3)
    parser.add_argument('--prompt_text', type=str, default='{"soft"}{"placeholder":"text_a"}{"mask"}.')
    parser.add_argument("--num_classes", type=int, default=2)
    
    # hyperparams
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)

    # optimizer 
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # steps
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()
    print(args.__dict__)

    set_seed(args.seed)

    plm, tokenizer, model_config, wrapper_class = load_plm(args.base_plm_family, args.base_plm_path)
    metadata, train_raw_dataset, eval_raw_dataset, test_raw_dataset = load_dataset(args)
    model = get_model(args.task, args.model)(args, plm, metadata, tokenizer, model_config, wrapper_class)
    train_dataloader, valid_dataloader, test_dataloader = get_tokenized_dataloader(train_raw_dataset, eval_raw_dataset, test_raw_dataset, model.tokenizer, model.template, model.wrapper_class)

    trainer = Trainer(args, model)
    if args.do_train:
        trainer.train(train_dataloader, valid_dataloader)
    
    if args.do_eval:
        trainer.eval(test_dataloader)
