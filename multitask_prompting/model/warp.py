from torch import nn
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer, MixedTemplate

class WARP(nn.Module):
    def __init__(self, args, plm, tune_plm, metadata): 
        self.plm = plm
        self.tune_plm = tune_plm
        self.verbalizer = SoftVerbalizer(
            classes=metadata['classes'],
            plm=self.plm,
            tokenizer = self.tokenizer
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
