import wandb
import torch
from transformers import  AdamW, Adafactor, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
from tqdm import tqdm
import time
class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if args.wandb:
            wandb.init(project=args.project_name, config = args)

        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_dataloader, valid_dataloader):
        if self.args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
            no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in self.model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer1 = AdamW(optimizer_grouped_parameters1, lr=self.args.learning_rate)
            scheduler1 = get_linear_schedule_with_warmup(
                optimizer1, 
                num_warmup_steps=self.args.warmup, num_training_steps=self.args.epochs)
        else:
            optimizer1 = None
            scheduler1 = None


        optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]},{'params': prompt_model.verbalizer.group_parameters_1},{'params': prompt_model.verbalizer.group_parameters_2}] # note that you have to remove the raw_embedding manually from the optimization
        if self.args.optimizer.lower() == "adafactor":
            optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                                    lr=self.args.learning_rate,
                                    relative_step=False,
                                    scale_parameter=False,
                                    warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=self.args.warmup) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        elif self.args.optimizer.lower() == "adamw":
            optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.args.learning_rate) # usually lr = 0.5
            scheduler2 = get_linear_schedule_with_warmup(
                            optimizer2, 
                            num_warmup_steps=self.args.warmup, num_training_steps=self.args.epochs) # usually num_warmup_steps is 500

        loss_func = torch.nn.CrossEntropyLoss()

        tot_loss = 0 
        log_loss = 0
        best_val_acc = 0
        glb_step = 0
        actual_step = 0
        leave_training = False

        acc_traces = []
        tot_train_time = 0
        self.model.train()
    
        pbar = tqdm(total=self.args.epochs, desc="Train epochs")
        pbar_update_freq = 10
        for epoch in range(self.args.epochs):
            # print(f"Begin epoch {epoch}")
            ebar = tqdm(total=len(train_dataloader), desc="Epoch progress")
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(self.device)
                tot_train_time -= time.time()
                logits = self.model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                actual_step += 1

                if actual_step % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    glb_step += 1
                    if glb_step % pbar_update_freq == 0:
                        aveloss = (tot_loss - log_loss)/pbar_update_freq
                        # pbar.update(10)
                        pbar.set_postfix({'loss': aveloss})

                        metrics = {"average_loss_10_steps": aveloss}
                    
                        if self.args.wandb:
                            wandb.log(metrics)

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
                ebar.update(1)

                if actual_step % self.args.gradient_accumulation_steps == 0 and glb_step >0 and glb_step % self.args.eval_every == 0:
                    val_acc = self.evaluate(valid_dataloader)
                    if val_acc >= best_val_acc:
                        # TODO Model saving code here if wandb is not already taking care of it
                        # torch.save(prompt_model.state_dict(),f"{args.project_root}/../ckpts/{this_run_unicode}.ckpt")
                        best_val_acc = val_acc
                    
                    acc_traces.append(val_acc)

                    metrics = {"validation_accuracy": val_acc}
                    
                    if self.args.wandb:
                        wandb.log(metrics)
                    
                    tqdm.write(f'Glb_step {glb_step}, val_acc {val_acc}, average time {tot_train_time/actual_step}')
                    self.model.train()
            ebar.close()
            pbar.update(1)
                # if glb_step > self.args.max_steps:
                #     leave_training = True
                #     break
        pbar.close()
            # if leave_training:
            #     break 
    
    def evaluate(self, dataloader):
        self.model.eval()
        allpreds = []
        alllabels = []
    
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc
