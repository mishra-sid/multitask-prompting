import torch
import torch.nn as nn
from openprompt.prompts import SoftVerbalizer, MixedTemplate

class SoftVerbalizerPLMedInit(SoftVerbalizer):
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                ):
        super()._init_(tokenizer=tokenizer, num_classes=num_classes, classes=classes,model=model,classes = classes, label_words=label_words,prefix=prefix,multi_token_handler=multi_token_handler)
        print("SoftVerbalizerPLMedInit called")

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        words_ids = []
        for word in self.label_words:
            if isinstance(word, list):
                logger.warning("Label word for a class is a list, only use the first word.")
            word = word[0]
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning("Word {} is split into multiple tokens: {}. \
                    If this is not what you expect, try using another word for this verbalizer" \
                    .format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)

        max_len  = max([len(ids) for ids in words_ids])
        words_ids_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in words_ids]
        words_ids = [ids+[0]*(max_len-len(ids)) for ids in words_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)


        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)

        # init_data = self.original_head_last_layer[self.label_words_ids,:]*self.label_words_mask.to(self.original_head_last_layer.dtype).unsqueeze(-1)
        init_data = self.plm(self.label_words_ids,self.label_words_mask, output_hidden_states = True).hidden_states[-1]*self.label_words_mask.to(self.original_head_last_layer.dtype).unsqueeze(-1)
        init_data = init_data.sum(dim=1)/self.label_words_mask.sum(dim=-1,keepdim=True)

        # get initialization by passing label through plm


        if isinstance(self.head, torch.nn.Linear):
            self.head.weight.data = init_data
            self.head.weight.data.requires_grad=True
        else:
            '''
            getattr(self.head, self.head_last_layer_full_name).weight.data = init_data
            getattr(self.head, self.head_last_layer_full_name).weight.data.requires_grad=True # To be sure
            '''
            self.head_last_layer.weight.data = init_data
            self.head_last_layer.weight.data.requires_grad=True

