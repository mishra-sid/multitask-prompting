from torch import nn
class BERT(nn.Module):
        def __init__(self, args, plm, metadata, tokenizer, model_config, wrapper_class):
            super(BERT, self).__init__()
            self.args = args
            self.plm = plm
            self.tokenizer = tokenizer
            self.model_ci
            
                  
        def forward(self, input):
            embd = self.bert(input)
