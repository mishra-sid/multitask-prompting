import argparse
from email.policy import default
from trainer import Trainer

from utils import set_seed


from data.data_loader import load_and_cache_examples
from tokenizer import load_tokenizer 

if __name__ == "__name__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    
    # args
    parser.add_argument('--task', type='str', default='classification')
    parser.add_argument('--dataset', type='str')
    parser.add_argument('--model', type='str', default='bert')
    parser.add_argument('--base_plm', type='str', default='bert-base-uncased')
    parser.add_argument("--tune_plm", type=bool, default=False)

    # hyperparams
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--eval_every', type=int) 
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--warmup", type=int, default=500)

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()

  
    
    set_seed(args)
    tokenizer = load_tokenizer(args)

    metadata, train_dataset, eval_dataset, test_dataset = load_and_cache_examples(args, tokenizer)

    trainer = Trainer(args, metadata)
    if args.do_train:
        trainer.train(args, train_dataset)
    
    if args.do_eval:
        trainer.eval(args, eval_dataset, test_dataset)










