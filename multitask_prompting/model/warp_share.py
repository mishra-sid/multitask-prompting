from torch import nn
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer, MixedTemplate

class WARPShare(nn.Module):
    def __init__(self, args, plm, metadata, tokenizer, model_config, wrapper_class): 
        super(WARPShare, self).__init__()
        self.plm = plm
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.wrapper_class = wrapper_class
        
        self.verbalizers = {}
        self.templates = {}
        self.models = {}

        for scenario in metadata.keys():
            self.verbalizers[scenario] = SoftVerbalizer(
                classes=metadata[scenario]['classes'],
                label_words=metadata[scenario]['label_words'],
                model=self.plm,
                tokenizer = self.tokenizer
            )

            self.templates[scenario] =  MixedTemplate(
                tokenizer = self.tokenizer,
                text= args.prompt_text,
                model = self.plm
            )

            self.models[scenario] = PromptForClassification(
                template = self.templates[scenario],
                plm = self.plm,
                verbalizer = self.verbalizers[scenario]
            )
    
    def forward(self, scenario, inp):
        return self.models[scenario](inp)
