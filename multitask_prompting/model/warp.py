from torch import nn
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer, MixedTemplate

class WARP(nn.Module):
    def __init__(self, args, plm, metadata, tokenizer, model_config, wrapper_class): 
        print("new model initialized")
        super(WARP, self).__init__()
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
            self.verbalizers[scenario] = self.verbalizers[scenario].cuda()
            prompt_text = '{"soft": None, "duplicate":'+ str(args.prompt_num_soft_tokens)+'}{"placeholder":"text_a"}{"mask"}.'
            if  metadata[scenario]["text_num"]=="MULTIPLE" :
                prompt_text = '{"soft": None, "duplicate":'+ str(int((args.prompt_num_soft_tokens)/2)) +'}{"placeholder":"text_a"}{"soft": None, "duplicate":'+str(int((args.prompt_num_soft_tokens)/2))  +'}{"placeholder":"text_b"}{"mask"}.'

            self.templates[scenario] =  MixedTemplate(
                tokenizer = self.tokenizer,
                text=  prompt_text,
                model = self.plm
            )
            self.templates[scenario] = self.templates[scenario]

            self.models[scenario] = PromptForClassification(
                template = self.templates[scenario],
                plm = self.plm,
                verbalizer = self.verbalizers[scenario]
            )

            self.models[scenario] = self.models[scenario]

        
    def forward(self, inp, scenario):
        return self.models[scenario](inp)
