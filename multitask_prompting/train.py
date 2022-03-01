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
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1 )
    parser.add_argument("--pbar_update_freq", type = int, default = 10 )

    


    # hyperparams
    parser.add_argument('--lr_plm',default=0.01, type=float)
    parser.add_argument('--lr_prompt',default=0.5, type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--eval_every', type=int) 
    parser.add_argument("--optimizer_plm", type=str, default="adamw", help="optimizer for plm")
    parser.add_argument("--optimizer_prompt", type=str, default="adamw", help= "optimizer for prompt tuning parameters")
    parser.add_argument("--warmup_steps_plm", type=int, default=500)
    parser.add_argument("--warmup_steps_prompt", type=int, default=500)
    parser.add_argument("--warmup_tot_steps", type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default = 0.0)
    parser.add_argument('--max_norm',default=1.0, type=float)


    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()

  
    
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)

    metadata, train_dataloader, eval_dataloader, test_dataloader = load_and_cache_examples(args, tokenizer)

    trainer = Trainer(args, metadata)
    if args.do_train:
        trainer.train(train_dataloader,eval_dataloader)
    
    if args.do_eval:
        trainer.eval(test_dataloader)










