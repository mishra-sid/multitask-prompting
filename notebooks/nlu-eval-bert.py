import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plm', type=str, default='bert-base-cased')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
device  = args.device

from datasets import load_dataset

data_set = load_dataset("nlu_evaluation_data",split='train')
data_set = data_set.shuffle(seed=42)

labels = data_set.features["label"].names
num_labels = len(labels)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(args.plm)
model = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=num_labels).to(device)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=64, truncation=True, return_tensors="pt")

data_set = data_set.map(tokenize_function)
data_set = data_set.rename_column("label", "labels")
data_set.set_format('torch', columns=["input_ids", "attention_mask", "labels"])
train_valid_test = data_set.train_test_split(test_size=0.3)
train_valid = train_valid_test['train'].train_test_split(test_size=0.3)

train_data =  train_valid['train']
valid_data = train_valid['test']
test_data = train_valid_test['test']

import numpy as np
from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
from transformers import get_scheduler

num_epochs = 20
batch_size = 32
num_training_steps = num_epochs * int( len(train_data) / batch_size )
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
eval_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

print(f"Number of parameters {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

from datasets import load_metric

def evaluate(dataloader):
    metric = load_metric("accuracy")
    model.eval()
    for batch in dataloader:
        gbatch = {}
        for k, v in batch.items():
            if k in ["input_ids", "attention_mask"]:
                gbatch[k] = v[0].to(device)
            elif k == "labels":
                gbatch[k] = v.to(device) 

        
        with torch.no_grad():
            outputs = model(**gbatch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

best_yet = 0.0
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        gbatch = {}
        for k, v in batch.items():
            if k in ["input_ids", "attention_mask"]:
                gbatch[k] = v[0].to(device)
            elif k == "labels":
                gbatch[k] = v.to(device) 
        outputs = model(**gbatch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    metrics = evaluate(eval_dataloader)
    print(f"epoch {epoch} metrics {metrics}")
    if metrics['accuracy'] > best_yet:
        metrics = evaluate(test_dataloader)
        print(f"test acc: {metrics}")