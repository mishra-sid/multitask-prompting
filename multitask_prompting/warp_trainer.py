import torch
import argparse
import os
import wandb
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets
from openprompt.data_utils import InputExample
from openprompt.prompts import MixedTemplate
from openprompt import PromptForClassification


parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--use_cuda",action="store_false" )
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1 )

parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--model", type=str, default='bert', help="Using bert-base-cased")
parser.add_argument("--model_name_or_path", default='bert-base-cased')
parser.add_argument("--project_root", default="./", help="The project root in the file system")

#TODO data_dir not used currently
parser.add_argument("--data_dir", type=str, default="../data") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 

parser.add_argument("--result_file", type=str, default="results/results.txt")
parser.add_argument("--max_steps", default=100, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--eval_every_steps", type=int, default=5)

# TODO add similar hyper parameter for the mixed prompt used
# parser.add_argument("--soft_token_num", type=int, default=20)

parser.add_argument("--optimizer", type=str, default="adamw")
args = parser.parse_args()

wandb.init(config = args, project  = 'trial_prompting', entity = 'trial_1')

args.result_file = os.path.join(args.project_root, args.result_file)

content_write = "="*20+"\n"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"n_epochs: {args.n_epochs}\t"

content_write += "\n"

print(content_write)

import random

#For saving the run:
#this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)


#Loading Dataset 
data = load_dataset("nlu_evaluation_data",split='train')
data_alarm= data.filter(lambda example: example['scenario'] == 'alarm')
data_audio= data.filter(lambda example: example['scenario'] == 'audio')

data_alarm = data_alarm.train_test_split(test_size=0.2)
data_audio = data_audio.train_test_split(test_size=0.2)

data_combined_train = concatenate_datasets([data_alarm['train'], data_audio['train']]).shuffle(seed = 42)
data_combined_test = concatenate_datasets([data_alarm['test'], data_audio['test']]).shuffle(seed = 42)


classes = [0,1,2,3,4,5,6]
dataset_combined_train = []

for i in range(len(data_combined_train)):
    example = data_combined_train[i]
    e = InputExample(guid = i, text_a= example['text'], label = example['label'])
    dataset_combined_train.append(e)

dataset_combined_test = []

for i in range(len(data_combined_test)):
    example = data_combined_test[i]
    e = InputExample(guid = i, text_a= example['text'], label = example['label'])
    dataset_combined_test.append(e)


from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

from openprompt.prompts import SoftVerbalizer
SoftpromptVerbalizer = SoftVerbalizer(
    classes = classes,
    plm=plm,
    tokenizer = tokenizer
)



WpromptTemplate =  MixedTemplate(    
    tokenizer = tokenizer,
    text=' {"soft"} {"soft"} {"placeholder":"text_a"} {"soft"} {"soft"} {"mask"}.',
    model = plm
)

wrapModel = PromptForClassification(
    template = WpromptTemplate,
    plm = plm,
    verbalizer = SoftpromptVerbalizer
)

from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(
    dataset = dataset_combined_train,
    tokenizer = tokenizer,
    template = WpromptTemplate,
    tokenizer_wrapper_class=WrapperClass
)

from openprompt import PromptDataLoader
validation_dataloader = PromptDataLoader(
    dataset = dataset_combined_test,
    tokenizer = tokenizer,
    template = WpromptTemplate,
    tokenizer_wrapper_class=WrapperClass
)

from tqdm import tqdm
import time
prompt_model = wrapModel.cuda()

wandb.watch(prompt_model)

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
   
    for step, inputs in enumerate(dataloader):
        if args.use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
loss_func = torch.nn.CrossEntropyLoss()

tot_step = args.max_steps


if args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=500, num_training_steps=tot_step)
else:
    optimizer1 = None
    scheduler1 = None


optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
if args.optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                            lr=args.prompt_lr,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif args.optimizer.lower() == "adamw":
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr) # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
                    optimizer2, 
                    num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500


tot_loss = 0 
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False

acc_traces = []
tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()

pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(args.n_epochs):
    print(f"Begin epoch {epoch}")
    for step, inputs in enumerate(train_dataloader):
        if args.use_cuda:
            inputs = inputs.cuda()
        tot_train_time -= time.time()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1

        if actual_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

        
        if optimizer1 is not None:
            optimizer1.step()
            optimizer1.zero_grad()
        if scheduler1 is not None:
            scheduler1.step()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

        tot_train_time += time.time()

        if actual_step % args.gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
            val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
            if val_acc >= best_val_acc:
                # TODO Model saving code here if wandb is not already taking care of it
                # torch.save(prompt_model.state_dict(),f"{args.project_root}/../ckpts/{this_run_unicode}.ckpt")
                best_val_acc = val_acc
            
            acc_traces.append(val_acc)

            metrics = {"validation_accuracy": val_acc}
            wandb.log(metrics)
            print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
            prompt_model.train()

        if glb_step > args.max_steps:
            leave_training = True
            break
    
    if leave_training:
        break  

thres99 = 0.99*best_val_acc
thres98 = 0.98*best_val_acc
thres100 = best_val_acc
step100=step98=step99=args.max_steps
for val_time, acc in enumerate(acc_traces):
    if acc>=thres98:
        step98 = min(val_time*args.eval_every_steps, step98)
        if acc>=thres99:
            step99 = min(val_time*args.eval_every_steps, step99)
            if acc>=thres100:
                step100 = min(val_time*args.eval_every_steps, step100)


content_write = ""
content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
content_write += "\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)
