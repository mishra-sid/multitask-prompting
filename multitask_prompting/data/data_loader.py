
from datasets import load_dataset
from openprompt import PromptDataLoader
from torch.utils.data import DataLoader
from openprompt.prompts import MixedTemplate
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer


def load_and_cache_examples(args,tokenizer_dictionary):
    data_set = load_dataset(args.dataset).shuffle(seed=42)
    classes = data_set.features['label'].names
    data_set = data_set.rename_column("label", "labels")

    if args.tune_plm is "prompting" :
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


        SoftpromptVerbalizer = SoftVerbalizer(
            classes = classes,
            plm=tokenizer["plm"],
            tokenizer = tokenizer_dictionary["tokenizer"]
        )
        WpromptTemplate =  MixedTemplate(
            tokenizer = tokenizer_dictionary["tokenizer"],
            text= args.prompt_text
            model = tokenizer_dictionary["plm"]
        )

        wrapModel = PromptForClassification(
            template = WpromptTemplate,
            plm = tokenizer_dictionary["plm"],
            verbalizer = SoftpromptVerbalizer
        )
        train_dataloader = PromptDataLoader(
            dataset = dataset_train,
            tokenizer = tokenizer_dictionary["tokenizer"],
            template = WpromptTemplate,
            tokenizer_wrapper_class=tokenizer_dictionary["WrapperClass"]
        )

        validation_dataloader = PromptDataLoader(
            dataset = dataset_valid,
            tokenizer = tokenizer_dictionary["tokenizer"],
            template = WpromptTemplate,
            tokenizer_wrapper_class=tokenizer["WrapperClass"]
        )
        test_dataloader = PromptDataLoader(
            dataset = dataset_test,
            tokenizer = tokenizer_dictionary["tokenizer"],
            template = WpromptTemplate,
            tokenizer_wrapper_class=tokenizer["WrapperClass"]
        )
    else:
        tokenized_datasets = data_set.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        test_train_dataset = tokenized_datasets.train_test_split(test_size=args.test_split)
        train_valid_dataset =test_train_dataset["train"].train_test_split(test_size=args.valid_split)
        train_dataloader = DataLoader(train_valid_dataset["train"], shuffle=True, batch_size=args.batch_size)
        valid_dataloader = DataLoader(train_valid_dataset["test"], shuffle=True, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_train_dataset["test"], batch_size=args.batch_size)
    metadata={
        "class_labels":classes
    }

    return (metadata,train_dataloader,valid_dataloader,test_dataloader)
#
# train_dataloader = PromptDataLoader(
#     dataset = dataset_combined_train,
#     tokenizer = tokenizer,
#     template = WpromptTemplate,
#     tokenizer_wrapper_class=WrapperClass
# )
#
# validation_dataloader = PromptDataLoader(
#     dataset = dataset_combined_test,
#     tokenizer = tokenizer,
#     template = WpromptTemplate,
#     tokenizer_wrapper_class=WrapperClass
# )

def tokenize_function(dataset):
    return tokenizer["tokenizer"](dataset["text"], padding="max_length", truncation=True)


