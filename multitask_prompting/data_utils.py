import datasets
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from pathlib import Path

def load_dataset(args):
    data_set = datasets.load_dataset(args.dataset)['train'].shuffle(seed=args.seed)
    classes = data_set.features['label'].names
    data_set = data_set.rename_column("label", "labels")
    metadata = {'classes': classes}

    trainvalid_test_dataset = data_set.train_test_split(test_size=args.test_split)
    train_valid_dataset =trainvalid_test_dataset["train"].train_test_split(test_size=args.valid_split)
    dataset_train=[]
    for i in range(len(train_valid_dataset["train"])):
        entry = train_valid_dataset["train"][i]
        e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
        dataset_train.append(e)

    dataset_test = []
    for i in range(len(trainvalid_test_dataset["test"])):
        entry = trainvalid_test_dataset["test"][i]
        e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
        dataset_test.append(e)

    dataset_valid = []
    for i in range(len(train_valid_dataset["test"])):
        entry = train_valid_dataset["test"][i]
        e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
        dataset_valid.append(e)

    
    return metadata, dataset_train, dataset_valid, dataset_test

def get_tokenized_dataloader(args, train_raw_dataset, eval_raw_dataset, test_raw_dataset, tokenizer, template, wrapper_class):
    train_dataloader = PromptDataLoader(
        dataset = train_raw_dataset,
        tokenizer = tokenizer,
        template = template,
        tokenizer_wrapper_class=wrapper_class,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    eval_dataloader = PromptDataLoader(
        dataset = eval_raw_dataset,
        tokenizer = tokenizer,
        template = template,
        tokenizer_wrapper_class=wrapper_class,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )
    test_dataloader = PromptDataLoader(
        dataset = test_raw_dataset,
        tokenizer = tokenizer,
        template = template,
        tokenizer_wrapper_class=wrapper_class,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    return train_dataloader, eval_dataloader, test_dataloader
