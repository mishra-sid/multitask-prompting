import datasets

from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from pathlib import Path


def load_nlu_dataset(args):
    data_set = datasets.load_dataset(args.dataset)['train'].shuffle(seed=args.seed)
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
        metadata[scenario]['text_num']= "SINGLE"

        
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
                e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
                dataset_train.append(e)
            dataset_train_all.extend(dataset_train)

            dataset_test = []
            for i in range(len(trainvalid_test_dataset["test"])):
                entry = trainvalid_test_dataset["test"][i]
                e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
                dataset_test.append(e)
            dataset_test_all.extend(dataset_test)

            dataset_valid = []
            for i in range(len(train_valid_dataset["test"])):
                entry = train_valid_dataset["test"][i]
                e = InputExample(guid = i, text_a= entry['text'], label = entry['labels'])
                dataset_valid.append(e)
            dataset_valid_all.extend(dataset_valid)
        raw_datasets[scenario] = {'train': dataset_train_all, 'valid': dataset_valid_all, 'test': dataset_test_all }
    
    return metadata, raw_datasets

def get_tokenized_dataloader(args, metadata, raw_datasets, tokenizer, templates, wrapper_class):
    dataloaders = {}
    
    for scenario in metadata.keys(): 
        dataloaders[scenario] = {}
        dataloaders[scenario]['train'] = PromptDataLoader(
            dataset = raw_datasets[scenario]['train'],
            tokenizer = tokenizer,
            template = templates[scenario],
            tokenizer_wrapper_class=wrapper_class,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )

        dataloaders[scenario]['valid'] = PromptDataLoader(
            dataset =  raw_datasets[scenario]['valid'],
            tokenizer = tokenizer,
            template = templates[scenario],
            tokenizer_wrapper_class=wrapper_class,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
        dataloaders[scenario]['test'] = PromptDataLoader(
            dataset =  raw_datasets[scenario]['test'],
            tokenizer = tokenizer,
            template = templates[scenario],
            tokenizer_wrapper_class=wrapper_class,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )

    return dataloaders
