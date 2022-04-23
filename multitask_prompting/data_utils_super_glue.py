import datasets


from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from pathlib import Path
from openprompt.data_utils import PROCESSORS

configs = ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc']
# no handling for axb axg and record is to be handled differently

def load_super_glue_dataset(args):
    multiple_inputs={}
    scenarios = set(configs)
    metadata = {}
    raw_datasets = {}
    for scenario in configs:
        Processor = PROCESSORS[args.dataset + "." + scenario]
        all_classes =Processor().get_labels()
        metadata[scenario] = { 'classes': all_classes }
        if args.verbalizer_init == 'random':
            metadata[scenario]['label_words'] = None
        elif args.verbalizer_init == "raw":
            metadata[scenario]['label_words'] = {l: l for l in all_classes}

        dataset_train_all =  Processor().get_train_examples(args.data_dir)
        dataset_valid_all = Processor().get_dev_examples(args.data_dir)
        dataset_test_all = Processor().get_test_examples(args.data_dir)

        if  scenario in ["copa", "wsc"]:
           metadata[scenario]['text_num']= "SINGLE" 
        else:
           metadata[scenario]['text_num']= "MULTIPLE"
        raw_datasets[scenario] = {'train': dataset_train_all, 'valid': dataset_valid_all, 'test': dataset_test_all }

    return metadata,raw_datasets
