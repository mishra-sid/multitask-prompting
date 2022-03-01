# -*- coding: utf-8 -*-
"""bert-base

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KoYD0Gtvx8uk6wPYdrfGAwf4b2o7IVVZ
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import os

os.chdir('./drive/MyDrive/696ds')

!pip install transformers

from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-cased')

!pip install datasets

import torch
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets

data_set = load_dataset("nlu_evaluation_data",split='train')

data_alarm_audio= data_set.filter(lambda example: example['scenario'] == 'alarm' or example['scenario'] == 'audio')

data_alarm_audio

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoTokenizer,DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = data_alarm_audio.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tt_set = tokenized_datasets.train_test_split(test_size=0.3)

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(
#     tt_set["train"], shuffle=True, batch_size=8, collate_fn=data_collator
# )
# eval_dataloader = DataLoader(
#     tt_set["test"], batch_size=8, collate_fn=data_collator
# )

tt_set

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=7)

!pip install wandb

from transformers import TrainingArguments

# training_args = TrainingArguments("test_trainer")
training_args = TrainingArguments(output_dir = "checkpoints-2",per_device_train_batch_size=2,save_steps=20,report_to="wandb")

from transformers import Trainer

trainer = Trainer(model=model, args=training_args, train_dataset=tt_set["train"], eval_dataset=tt_set["test"])

trainer.train()

"""**STARTING WITH HYPER PARAMETER TUNING USING WANDB**"""

import wandb

sweep_config ={
    'method' : 'random'
}

metric = {
    'name' : 'lose',
    'goal' : 'minimize'
}

sweep_config['metric'] = metric
# def train_bert():
#    configs = {
#        'layers': 128
#    }
   
#    config = wandb.config
#    config.epochs = 5

parameters_dict = {
    'batch_size': {
        'values': [2,3,4]
        },
    'learning_rate': {
        'values': [.001,.01]
        },

    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': [10,20,30]}
    })

sweep_id = wandb.sweep(sweep_config, project="bertbase_test")

def build_dataset(batch_size):
    output=[tt_set["train"][i:i + batch_size] for i in range(0, len(tt_set["train"]), batch_size)]
    return output

def train_epoch( loader,test_set):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=7)
    trainer = Trainer(model=model, args=training_args, train_dataset=loader, eval_dataset=test_set)
    trainer.train()
    return ""

from transformers import Trainer

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=sweep_config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        print("config",config)
        # loader = build_dataset(config.batch_size)
      #  network = build_network(config.fc_layer_size, config.dropout)
        # optimizer = build_optimizer(config.optimizer, config.learning_rate)
        training_args = TrainingArguments(output_dir = "checkpoints-2",num_train_epochs = config["epochs"],per_device_train_batch_size=config["batch_size"],learning_rate = config["learning_rate"],save_steps=20)
        model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=7)
        trainer2 = Trainer(model=model2, args=training_args, train_dataset=tt_set["train"], eval_dataset=tt_set["test"])
        trainer2.train()
        # avg_loss = trainer.compute_loss
        # wandb.log({"loss": avg_loss, "epoch": config.epochs}) 
        # # for epoch in range(config.epochs):
        #     avg_loss = train_epoch( loader,tt_set["test"])

wandb.agent(sweep_id, train, count=5)
