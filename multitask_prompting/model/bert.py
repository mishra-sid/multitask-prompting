
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

class BERT(BertPreTrainedModel):
        def __init__(self, metadata):    
            self.bert = BertModel(config=metadata) 
            
        def forward(self, input):
            embd = self.bert(input)
            
