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

all_labels = data_set.features['label'].names
all_scens = list(map(lambda x: x.split('_')[0], all_labels))
scenarios = set(all_scens)
scenarios = set(filter(lambda x : all_scens.count(x) > 1, scenarios))
# scenarios = {'alarm', 'audio'}

metadata = {}
raw_datasets = {}

for scenario in scenarios:
    data_set_scenario = data_set.filter(lambda ex: ex['scenario'] == scenario)
    classes = list(filter(lambda x: x.split('_')[0] == scenario, all_labels))
    min_label = list(map(lambda x: x.split('_')[0] == scenario, all_labels)).index(True)
            
    def change_labels(ex):
        ex['label'] = ex['label'] - min_label
        return ex
    
    data_set_scenario = data_set_scenario.map(change_labels)
    data_set_scenario = data_set_scenario.rename_column("label", "labels")

    metadata[scenario] = { 'classes': classes } 
    
    if args.verbalizer_init == 'random':
        metadata[scenario]['label_words'] = None
    elif args.verbalizer_init == "raw":
        metadata[scenario]['label_words'] = {l: l for l in classes}
    elif args.verbalizer_init == "first":
        metadata[scenario]['label_words'] = {l: l.split("_")[0] for l in classes}
    elif args.verbalizer_init == "last":
        metadata[scenario]['label_words'] = {l: l.split("_")[-1] for l in classes}

    dataset_train_all=[]
    dataset_valid_all=[]
    dataset_test_all=[]
    
    for label in classes:
        data_set_scenario_label = data_set_scenario.filter(lambda ex: all_labels[min_label + ex['labels']] == label)
        trainvalid_test_dataset = data_set_scenario_label.train_test_split(test_size=args.test_split)
        train_valid_dataset =trainvalid_test_dataset["train"].train_test_split(test_size=args.valid_split)
        dataset_train=[]
        for i in range(len(train_valid_dataset["train"])):
            entry = train_valid_dataset["train"][i]
            dataset_train.append(entry)
        dataset_train_all.extend(dataset_train)

        dataset_test = []
        for i in range(len(trainvalid_test_dataset["test"])):
            entry = trainvalid_test_dataset["test"][i]
            dataset_test.append(entry)
        dataset_test_all.extend(dataset_test)

        dataset_valid = []
        for i in range(len(train_valid_dataset["test"])):
            entry = train_valid_dataset["test"][i]
            dataset_valid.append(entry)
        dataset_valid_all.extend(dataset_valid)
    raw_datasets[scenario] = {'train': dataset_train_all, 'valid': dataset_valid_all, 'test': dataset_test_all }

tokenized_datasets = {}

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(args.plm)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=64, truncation=True, return_tensors="pt")

import numpy as np
from datasets import load_metric
from transformers import AdamW
from transformers import get_scheduler

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

valid_accs = []
test_accs = []

for scenario in metadata.keys():
    tokenized_datasets[scenario] = {}
    for split in ['train', 'valid', 'test']:
        data_set = raw_datasets[scenario][split].map(tokenize_function)
        data_set = data_set.rename_column("label", "labels")
        data_set.set_format('torch', columns=["input_ids", "attention_mask", "labels"])
        tokenized_datasets[scenario][split] = data_set

    model = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
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