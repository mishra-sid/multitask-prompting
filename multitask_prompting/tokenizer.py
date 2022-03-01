from transformers import AutoTokenizer,DataCollatorWithPadding
from openprompt.plms import load_plm




def load_tokenizer(args):
    plm=NULL
    tokenizer=NULL
    model_config=NULL
    WrapperClass=NULL
    if(args.tune_plm is "prompting":
     plm, tokenizer,_, WrapperClass  =  load_plm(args.model, args.base_plm)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_plm)

    return {
    "tokenizer":tokenizer,
    "plm":plm,
    "WrapperClass":WrapperClass

    }

