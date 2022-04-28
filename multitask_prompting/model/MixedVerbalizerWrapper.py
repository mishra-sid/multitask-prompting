from openprompt.prompts import SoftVerbalizer, MixedTemplate
from inspect import Parameter
import json
from os import stat
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput
from transformers.models.t5 import  T5ForConditionalGeneration

class MixedVerbalizerWrapper(SoftVerbalizer):
    r"""
    The implementation of the verbalizer in `WARP <https://aclanthology.org/2021.acl-long.381/>`_
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 global_verbalizer =None,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                ):
        super(SoftVerbalizer, self).__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler

        head_name = [n for n,c in model.named_children()][-1]
        logger.info(f"The LM head named {head_name} was retrieved.")
        self.head = copy.deepcopy(getattr(model, head_name))
        max_loop = 5
        if not isinstance(self.head, torch.nn.Linear):
            module = self.head
            found = False
            last_layer_full_name = []
            for i in range(max_loop):
                last_layer_name = [n for n,c in module.named_children()][-1]
                last_layer_full_name.append(last_layer_name)
                parent_module = module
                module = getattr(module, last_layer_name)
                if isinstance(module, torch.nn.Linear):
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Can't not retrieve a linear layer in {max_loop} loop from the plm.")
            self.original_head_last_layer = module.weight.data
            self.hidden_dims = self.original_head_last_layer.shape[-1]
            self.head_last_layer_full_name = ".".join(last_layer_full_name)
            self.head_last_layer = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)
            setattr(parent_module, last_layer_name, self.head_last_layer)
        else:
            self.hidden_dims = self.head.weight.shape[-1]
            self.original_head_last_layer = getattr(model, head_name).weight.data
            self.head = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)


        if label_words is not None: # use label words as an initialization
            self.label_words = label_words


        if global_verbalizer != None:
            self.initialize_verbalizer(global_verbalizer)

    
    def initialize_verbalizer(self, global_verbalizer):
        init_data = global_verbalizer.head_last_layer.weight.data.detach().clone()

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

