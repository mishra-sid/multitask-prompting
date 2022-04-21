from collections import defaultdict
import wandb
import torch
import numpy as np
from transformers import  AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup 
from tqdm import tqdm
import time
from pathlib import Path
import json
from multitask_prompting import utils

class Trainer:
    def __init__(self, args, metadata, model):
        self.args = args
        self.metadata = metadata
        self.model = model

        if args.wandb:
            wandb.init(project=args.project_name, config = args)

        self.device = torch.device(args.device)
        self.model.to(self.device)
    
    def train(self, dataloaders):
        best_avg_val_acc = 0
        tot_train_time = 0
        self.model.train()
        best_val_accs = defaultdict(lambda : 0)
        test_accs = defaultdict(lambda : 0)
        val_accs = defaultdict(lambda : 0)

        optimizer1, optimizer2, scheduler1, scheduler2 = {},{},{},{} 
        num_training_steps = {}
        
        for scenario in self.metadata.keys():
            train_dataloader = dataloaders[scenario]['train']
            num_training_steps[scenario] = len(train_dataloader) * self.args.epochs
            if self.args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
                no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
                optimizer_grouped_parameters1 = [
                    {'params': [p for n, p in self.model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': self.args.weight_decay},
                    {'params': [p for n, p in self.model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optimizer1[scenario] = AdamW(optimizer_grouped_parameters1, lr=self.args.learning_rate)
                scheduler1[scenario] = get_linear_schedule_with_warmup(optimizer1[scenario], num_warmup_steps=self.args.warmup, num_training_steps=num_training_steps)
            else:
                optimizer1[scenario] = None
                scheduler1[scenario] = None

            if self.args.model_type == "prompt":
                if self.args.tune_plm:
                    optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.templates[scenario].named_parameters() if 'raw_embedding' not in name]},{'params': self.model.verbalizers[scenario].group_parameters_1},{'params': self.model.verbalizers[scenario].group_parameters_2}]
                else:
                    optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.templates[scenario].named_parameters() if 'raw_embedding' not in name]},{'params': self.model.verbalizers[scenario].group_parameters_2}]
                if self.args.optimizer.lower() == "adafactor":
                    optimizer2[scenario] = Adafactor(optimizer_grouped_parameters2,  
                                            lr=self.args.prompt_learning_rate,
                                            relative_step=False,
                                            scale_parameter=False,
                                            warmup_init=False) 
                    scheduler2[scenario] = get_linear_schedule_with_warmup(optimizer2[scenario],  num_warmup_steps=self.args.warmup) 
                elif self.args.optimizer.lower() == "adamw":
                    optimizer2[scenario] = AdamW(optimizer_grouped_parameters2, lr=self.args.prompt_learning_rate)
                    scheduler2[scenario] = get_constant_schedule_with_warmup(optimizer2[scenario],  num_warmup_steps=self.args.warmup) 
        
        progress_bar = tqdm(range(sum(ntr for ntr in num_training_steps.values())))
        #For every epoch
        for epoch in range(self.args.epochs):
            #Train every scenario
            for scenario in self.metadata.keys():
                train_dataloader = dataloaders[scenario]['train']
                loss_func = torch.nn.CrossEntropyLoss()
                for step, inputs in enumerate(train_dataloader):
                    inputs = inputs.to(self.device)
                    tot_train_time -= time.time()
                    logits = self.model(inputs, scenario=scenario)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                    loss.backward()
                    
                    progress_bar.update(1)
                    if optimizer1[scenario] is not None:
                        optimizer1[scenario].step()
                        optimizer1[scenario].zero_grad()
                    if scheduler1[scenario] is not None:
                        scheduler1[scenario].step()
                    if optimizer2[scenario] is not None:
                        optimizer2[scenario].step()
                        optimizer2[scenario].zero_grad()
                    if scheduler2[scenario] is not None:
                        scheduler2[scenario].step()
                    del inputs
                    tot_train_time += time.time()
            
            #Evaluate every scenario
            for scenario in self.metadata.keys():
                uniq = utils.get_uniq_str(self.args)
                save_path_dir = Path(self.args.model_dir)/ uniq/ scenario
                save_path_dir.mkdir(parents=True, exist_ok=True)
                
                val_loss, val_acc = self.evaluate_per_scenario(dataloaders,scenario, "valid")
                if val_acc >= best_val_accs[scenario]:
                    best_val_accs[scenario] = val_acc
                    test_loss, test_acc = self.evaluate_per_scenario(dataloaders, scenario,"test")    
                    torch.save(self.model.state_dict(), save_path_dir / "model.pth")

                info_path = save_path_dir / "info.json"
                info = None
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                else:
                    info = { "params": utils.get_num_trainable_params(self.args, self.metadata.keys(), self.model), "metrics": []}
                
                metric = {'epoch': epoch + 1, 'test_acc': test_acc, 'val_acc': val_acc}
                info["metrics"].append(metric)

                test_accs[scenario]  = test_acc
                val_accs[scenario] = val_acc

                with open(info_path, 'w') as wf:
                    json.dump(info, wf)
            
            # Calculate and log the average metrics
            val_avg_acc = np.mean(np.array([val_accs[i] for i in val_accs]))
            test_avg_acc = np.mean(np.array([test_accs[i] for i in test_accs]))

            agg_info_path = Path(self.args.model_dir)/ uniq / "agg_info.json"
            agg_info = None
            if agg_info_path.exists():
                with open(agg_info_path) as f:
                    agg_info = json.load(f)
            else:
                agg_info = { "params": utils.get_num_trainable_params(self.args, self.metadata.keys(), self.model), "metrics": []}
            agg_metric = {'epoch': epoch + 1, 'val_avg_acc': val_avg_acc, 'test_avg_acc': test_avg_acc}
            agg_info["metrics"].append(agg_metric)

            with open(agg_info_path, 'w') as wf:
                    json.dump(agg_info, wf)
                    
            wandb_metrics = {'val_avg_acc': val_avg_acc, 'test_avg_acc': test_avg_acc }
            print("epoch", epoch, "metrics", wandb_metrics) 
            if self.args.wandb:
                wandb.log(wandb_metrics)
            self.model.train()
    
    def evaluate_per_scenario(self, dataloaders,scenario, dataloader_type):
        dataloader = dataloaders[scenario][dataloader_type]
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
                logits = self.model(inputs, scenario=scenario)  
                labels = inputs['label']
                avg_loss += loss_func(logits, labels).item()
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                del inputs
            acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            avg_loss /= cnt
            return avg_loss, acc


    def evaluate(self, dataloaders, dataloader_type):
        with torch.no_grad():
            self.model.eval()
            lst_avg_loss = []
            lst_accs = []    
            lst_cnts = []        
            for scenario in self.metadata.keys():
                dataloader = dataloaders[scenario][dataloader_type]
                allpreds = []
                alllabels = []    
                avg_loss = 0
                loss_func = torch.nn.CrossEntropyLoss()
                cnt = 0
                for step, inputs in enumerate(dataloader):
                    cnt += 1
                    inputs = inputs.to(self.device)
                    logits = self.model(inputs, scenario=scenario)    
                    labels = inputs['label']
                    avg_loss += loss_func(logits, labels).item()
                    alllabels.extend(labels.cpu().tolist())
                    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    del inputs
                acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                avg_loss /= cnt
                lst_cnts.append(len(allpreds))
                lst_avg_loss.append(avg_loss)
                lst_accs.append(acc)
            lst_cnts_np = np.array(lst_cnts)
            lst_avg_loss_np = np.array(lst_avg_loss)
            lst_accs_np = np.array(lst_accs)
            return np.mean(lst_avg_loss_np), np.mean(lst_accs_np), np.sum(lst_accs_np * lst_cnts_np) / np.sum(lst_cnts_np),  np.std(lst_accs_np)
    
    def test(self, args, model, dataloaders):
        with torch.no_grad():
            path = Path(args.model_dir) / utils.get_uniq_str(args) / "model.pth" 
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            lst_avg_loss = []
            lst_accs = []    
            lst_cnts = []           
            for scenario in self.metadata.keys():
                dataloader = dataloaders[scenario]["test"]
                allpreds = []
                alllabels = []    
                avg_loss = 0
                loss_func = torch.nn.CrossEntropyLoss()
                cnt = 0
                for step, inputs in enumerate(dataloader):
                    cnt += 1
                    inputs = inputs.to(self.device)
                    logits = self.model(inputs, scenario=scenario)    
                    labels = inputs['label']
                    avg_loss += loss_func(logits, labels).item()
                    alllabels.extend(labels.cpu().tolist())
                    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    del inputs
                acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                avg_loss /= cnt
                lst_cnts.append(len(allpreds))
                lst_avg_loss.append(avg_loss)
                lst_accs.append(acc)
            lst_cnts_np = np.array(lst_cnts)
            lst_avg_loss_np = np.array(lst_avg_loss)
            lst_accs_np = np.array(lst_accs)
            mean_acc, std_acc = np.mean(lst_accs_np), np.std(lst_accs_np)
            print(f"The total test accuracy across scenarios is {np.sum(lst_accs_np * lst_cnts_np) / np.sum(lst_cnts_np)}")
            print(f"The mean/std test accuracy of the best model across scenarios is {mean_acc}/{std_acc}")
