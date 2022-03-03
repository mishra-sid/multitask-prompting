from torch import nn
class BERT(nn.Module):
        def __init__(self, args, plm, tune_plm, metadata): 
            self.plm = plm
            self.tune_plm = tune_plm   
                  
        def forward(self, input):
            embd = self.bert(input)
