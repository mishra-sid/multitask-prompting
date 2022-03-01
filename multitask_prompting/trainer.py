
from utils import MODEL_CLASSES
import wandb
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

class Trainer:
    def __init__(self, args, metadata):
        self.args = args

        # model is the overall model while plm is the base plm used in the model
        self.model = MODEL_CLASSES[args.model](args,metadata)
        if self.args.task.lower()=='classification_with_prompting':
            self.plm = self.model.plm
        else:
            self.plm = self.model

        self.metadata = metadata

        if args.wandb:
            wandb.init(project_name=args.project_name)

        self.device = torch.device('cuda' if args.cuda else 'cpu')
    
    def train(self, train_dataloader):
        args = self.args

        # Whether to tune finetune the model/non-prompting parameters:
        if args.task.lower()=='classification' or args.tune_plm:
            no_decay = ['bias', 'LayerNorm.weight'] # settting no decay to biase and LayerNorm parameters according to openprompt sample script
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in self.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if args.optimizer_plm.lower()=='adamw':
                optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.plm_lr)
                scheduler1 = get_linear_schedule_with_warmup(
                    optimizer1, 
                    num_warmup_steps=args.warmup_steps_plm, num_training_steps=warmup_tot_steps)
            else:
                print("Incorrect optimizer. Optimizer may not be supported yet.")
                exit()
        else:
            optimizer1 = None
            scheduler1 = None

        if args.task.lower()=='classification_with_prompting':
            optimizer_grouped_parameters2 = [{'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
            if args.optimizer_prompt.lower() == "adafactor":
                optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                                        lr=args.prompt_lr,
                                        relative_step=False,
                                        scale_parameter=False,
                                        warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
                scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            elif args.optimizer_prompt.lower() == "adamw":
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
    
    def evaluate():
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

