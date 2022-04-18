import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plm', type=str, default='bert-base-cased')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
device  = args.device

from datasets import load_dataset, concatenate_datasets

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

    dataset_train_all=[]
    dataset_valid_all=[]
    dataset_test_all=[]
    
    for label in classes:
        data_set_scenario_label = data_set_scenario.filter(lambda ex: all_labels[min_label + ex['labels']] == label)
        trainvalid_test_dataset = data_set_scenario_label.train_test_split(test_size=0.3)
        train_valid_dataset =trainvalid_test_dataset["train"].train_test_split(test_size=0.3)
        dataset_train_all.append(train_valid_dataset["train"])
        dataset_test_all.append(trainvalid_test_dataset["test"])
        dataset_valid_all.append(train_valid_dataset["test"])
    dataset_train_all = concatenate_datasets(dataset_train_all)
    dataset_valid_all = concatenate_datasets(dataset_valid_all)
    dataset_test_all = concatenate_datasets(dataset_test_all)
    raw_datasets[scenario] = {'train': dataset_train_all, 'valid': dataset_valid_all, 'test': dataset_test_all }



from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(args.plm)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=64, truncation=True, return_tensors="pt")

import numpy as np
from datasets import load_metric
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



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


valid_accs = []
test_accs = []
num_params = 0
for scenario in metadata.keys():
    tokenized_dataset = {}
    for split in ['train', 'valid', 'test']:
        data_set = raw_datasets[scenario][split].map(tokenize_function)
        # data_set = data_set.rename_column("label", "labels")
        data_set.set_format('torch', columns=["input_ids", "attention_mask", "labels"])
        tokenized_dataset[split] = data_set

    model = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=len(metadata[scenario]["classes"])).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 20
    batch_size = 32
    num_training_steps = num_epochs * int( len(tokenized_dataset["train"]) / batch_size )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset["train"], batch_size=batch_size)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_dataset["valid"], batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(tokenized_dataset["test"], batch_size=batch_size)
    num_params += sum([p.numel() for p in model.parameters() if p.requires_grad])

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
        val_acc = evaluate(eval_dataloader)['accuracy']
        # print(f"scenario {scenario} epoch {epoch} val acc {val_acc}")
        if val_acc > best_yet:
            test_acc = evaluate(test_dataloader)['accuracy']
            # print(f"scenario {scenario} test acc: {test_acc}")
    valid_accs.append(val_acc)
    test_accs.append(test_acc)    
            

valid_accs = np.array(valid_accs)
test_accs = np.array(test_accs) 
print(f"Number of parameters {num_params}")

val_acc_mean, val_acc_std = np.mean(valid_accs), np.std(valid_accs)
test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)

print(f"Mean/std accuracies for valid data is {val_acc_mean}/{val_acc_std}")
print(f"Mean/std accuracies for test data is {test_acc_mean}/{test_acc_std}")




