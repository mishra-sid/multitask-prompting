
from torch import nn
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer, MixedTemplate
import ast

class WARP(nn.Module):
    def __init__(self, args, plm, metadata, tokenizer, model_config, wrapper_class):
        super(WARP, self).__init__()
        self.plm = plm
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.wrapper_class = wrapper_class
        self.label_words = None

        if self.args.verbalizer_init != None:
            self.label_words = ast.literal_eval(self.args.verbalizer_init)


        self.verbalizer = SoftVerbalizer(
            classes=metadata['classes'],
            tokenizer = self.tokenizer,
            label_words = self.label_words,
            model = self.plm
        )

        self.template =  MixedTemplate(
            tokenizer = self.tokenizer,
            text= args.prompt_text,
            model = self.plm
        )

        self.model = PromptForClassification(
            template = self.template,
            plm = self.plm,
            verbalizer = self.verbalizer
        )

    def forward(self, inp):
        return self.model(inp)



