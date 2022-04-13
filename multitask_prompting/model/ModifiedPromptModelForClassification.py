
from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup

class ModifiedPromptForClassification(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer



    def extract_at_avg_input_embed(outputs: torch.Tensor,batch):
        print(outputs)

        outputs = outputs[torch.where(batch['loss_ids']>=0)]
        outputs = outputs.unsqueeze(1)
        outputs  = torch.mean(outputs,dim=0)
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_avg_input_embed(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_avg_input_embed(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits
