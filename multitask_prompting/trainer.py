import wandb
import torch
from transformers import  AdamW, Adafactor, get_linear_schedule_with_warmup
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
        num_training_steps = len(train_dataloader) * self.args.epochs
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

        if self.args.
        optimizer_grouped_parameters2 = [ {'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]},{'params': self.model.verbalizer.group_parameters_1},{'params': self.model.verbalizer.group_parameters_2}]
        if self.args.optimizer.lower() == "adafactor":
            optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                                    lr=self.args.learning_rate,
                                    relative_step=False,
                                    scale_parameter=False,
                                    warmup_init=False) 
            scheduler2 = get_linear_schedule_with_warmup(optimizer2,  num_warmup_steps=self.args.warmup, num_training_steps=num_training_steps) 
        elif self.args.optimizer.lower() == "adamw":
            optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.args.learning_rate) # usually lr = 0.5
            scheduler2 = get_linear_schedule_with_warmup(optimizer2,  num_warmup_steps=self.args.warmup, num_training_steps=num_training_steps) 

        loss_func = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        
        acc_traces = []
        tot_train_time = 0
        self.model.train()
    
        for epoch in tqdm(range(self.args.epochs)):
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(self.device)
                tot_train_time -= time.time()
                logits = self.model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                
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
            
            val_loss, val_acc = self.evaluate(valid_dataloader)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
            
            acc_traces.append(val_acc)

            metrics = {"validation_loss": val_loss, "validation_accuracy": val_acc}
                
            if self.args.wandb:
                wandb.log(metrics)
            self.model.train()
    
    def evaluate(self, dataloader):
        self.model.eval()
        allpreds = []
        alllabels = []
        avg_loss = 0
        loss_func = torch.nn.CrossEntropyLoss()
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            avg_loss += loss_func(logits, labels)
                
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        avg_loss /= len(dataloader)
        return avg_loss, acc
