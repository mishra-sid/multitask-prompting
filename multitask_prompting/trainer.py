import wandb
import torch
from transformers import  AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup 
from tqdm import tqdm
import time
from pathlib import Path
import json
from multitask_prompting import utils

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if args.wandb:
            wandb.init(project=args.project_name, config = args)

        self.device = torch.device(args.device)
        self.model.to(self.device)
    
    def train(self, train_dataloader, valid_dataloader, test_dataloader):
        num_training_steps = self.args.max_steps
        if self.args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
            no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in self.model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer1 = AdamW(optimizer_grouped_parameters1, lr=self.args.learning_rate)
            scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=self.args.warmup, num_training_steps=num_training_steps)
        else:
            optimizer1 = None
            scheduler1 = None

        if self.args.model_type == "prompt":
            if self.args.tune_plm:
                optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]},{'params': self.model.verbalizer.group_parameters_1},{'params': self.model.verbalizer.group_parameters_2}]
            else:
                optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]},{'params': self.model.verbalizer.group_parameters_2}]
            if self.args.optimizer.lower() == "adafactor":
                optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                                        lr=self.args.prompt_learning_rate,
                                        relative_step=False,
                                        scale_parameter=False,
                                        warmup_init=False) 
                scheduler2 = get_constant_schedule_with_warmup(optimizer2,  num_warmup_steps=self.args.warmup) 
            elif self.args.optimizer.lower() == "adamw":
                optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.args.prompt_learning_rate)
                scheduler2 = get_constant_schedule_with_warmup(optimizer2,  num_warmup_steps=self.args.warmup) 

        loss_func = torch.nn.CrossEntropyLoss()

        best_val_acc = 0

        tot_train_time = 0
        self.model.train()
        
        pbar_update_freq = 10
        progress_bar = tqdm(total=self.args.max_steps, desc="Training")
        

        actual_step = 0
        total_loss = 0
        log_loss = 0
        glb_step = 0
        leave_training = False


        for epoch in range(self.args.num_total_steps):
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(self.device)
                tot_train_time -= time.time()
                logits = self.model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                total_loss += loss.item()
                actual_step += 1

                if actual_step % self.args.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    glb_step += 1

                    if glb_step % pbar_update_freq == 0:
                        ave_loss = (total_loss - log_loss)/pbar_update_freq
                        progress_bar.update(10)
                        progress_bar.set_postfix({'loss': ave_loss})
                        log_loss = total_loss

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

                if actual_step % self.args.gradient_accumulation_steps == 0 and glb_step >0 and glb_step % self.args.eval_every_steps == 0: 
                    uniq = utils.get_uniq_str(self.args)
                    save_path_dir = Path(self.args.model_dir) / uniq
                    save_path_dir.mkdir(parents=True, exist_ok=True)
                    
                    val_loss, val_acc = self.evaluate(valid_dataloader)
                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        torch.save(self.model.state_dict(), save_path_dir / "model.pth")

                    info_path = save_path_dir / "info.json"
                    info = None
                    if info_path.exists():
                        with open(info_path) as f:
                            info = json.load(f)
                    else:
                        info = { "params": utils.get_num_trainable_params(self.args, self.model), "metrics": []}
                    
                    metric = {'step': glb_step, 'val_acc': val_acc}
                    info["metrics"].append(metric) 
                    
                    with open(info_path, 'w') as wf:
                        json.dump(info, wf)
                    
                    wandb_metrics = {"validation_loss": val_loss, "validation_accuracy": val_acc}
                    print("step", glb_step, "metrics", metric, flush=True) 
                    
                    if self.args.wandb:
                        wandb.log(wandb_metrics)
                    self.model.train()
                
                if glb_step > self.args.max_steps:
                    leave_training = True
                    break

            if leave_training:
                break
    
    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            allpreds = []
            alllabels = []
            avg_loss = 0
            loss_func = torch.nn.CrossEntropyLoss()
            cnt = 0
            for step, inputs in enumerate(dataloader):
                cnt += 1
                inputs = inputs.to(self.device)
                logits = self.model(inputs)    
                labels = inputs['label']
                avg_loss += loss_func(logits, labels).item()
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            avg_loss /= cnt
            return avg_loss, acc
    
    def test(self, args, model, dataloader):
        with torch.no_grad():
            path = Path(args.model_dir) / utils.get_uniq_str(args) / "model.pth" 
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            allpreds = []
            alllabels = []
            loss_func = torch.nn.CrossEntropyLoss()
            for step, inputs in enumerate(dataloader):
                inputs = inputs.to(self.device)
                logits = self.model(inputs)    
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            print(f"The accuracy of the model on the dataset is {acc}")
