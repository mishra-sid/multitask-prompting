import os
import string
from openprompt.utils.logging import logger
from openprompt.data_utils.utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template

import torch
from torch import nn

class MixedTemplateWrapper(MixedTemplate):
    """Mixed of manual token, trainable token and trainable that initialized with given hard token
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
    """
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                 shared_embedding:self.shared_embedding
                ):

        super(MixedTemplate,self).__init__(tokenizer = tokenizer, placeholder_mapping = placeholder_mapping)

        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.text = text

    def get_default_soft_token_ids(self) -> List[int]:
        return self.soft_token_ids

    def prepare(self):
        num_soft_token = 0
        text = []
        soft_token_ids = []
        idx_mp = {}
        emb_mp = {}
        for d in self.text:
            if "soft" not in d and "soft_id" not in d:
                text.append(d)
                soft_token_ids.append(0)
                continue

            old_num = num_soft_token

            if "soft_id" in d:
                if not isinstance(d["soft_id"], int) or d["soft_id"] <= 0:
                    raise ValueError(f'soft_id should be integer greater than zero, but get {d["soft_id"]}')
                if d["soft_id"] in idx_mp:
                    id_list = idx_mp[d["soft_id"]]
                    text.extend([{"soft":None} for _ in range(len(id_list))])
                    soft_token_ids.extend(id_list)
                    continue
                else:
                    if "soft" not in d: d["soft"] = None

            if d["soft"] is None:
                if "duplicate" in d:
                    if "same" in d and d["same"]:
                        num_soft_token += 1
                        id_list = [num_soft_token for _ in range(len(d["duplicate"]))]
                    else:
                        num_soft_token += d["duplicate"]
                        id_list = list(range(old_num+1, num_soft_token+1))
                else:
                    num_soft_token += 1
                    id_list = [num_soft_token]
                text.extend([{"soft":""} for _ in range(len(id_list))])
            else:
                token_ids = self.tokenizer(d["add_prefix_space"] + d["soft"], add_special_tokens=False)["input_ids"]
                surface_forms = self.tokenizer.convert_ids_to_tokens(token_ids)
                assert len(token_ids) == len(surface_forms)
                num_soft_token += len(token_ids)
                id_list = list(range(old_num+1, num_soft_token+1))
                for idx, soft_id in enumerate(id_list):
                    emb_mp[soft_id] = token_ids[idx]

                text.extend([{"soft": surface_form} for surface_form in surface_forms])
            soft_token_ids.extend(id_list)

            if "soft_id" in d:
                idx_mp[d["soft_id"]] = id_list

        self.num_soft_token = num_soft_token
        self.text = text
        self.soft_token_ids = soft_token_ids

        # Generate the embedding needed for soft tokens

        scenario_specific_embedding =  nn.Embedding(1 + self.num_soft_token, self.embedding_size)
        self.soft_embedding = torch.concate((scenario_specific_embedding,self.shared_embedding),axis=0)
        print( self.soft_embedding.shape())

        for soft_id, token_id in emb_mp.items():
            self.soft_embedding.weight.data[soft_id, :] = self.raw_embedding.weight.data[token_id, :].clone().detach().requires_grad_(True)

        # if "post_processing" in d:
        #     if d["post_processing"] == "mlp":
        #         pass # TODO one mlp or more than one
        #     else:
        #         raise ValueError(f'post_processing of {d["post_processing"]} is not supported yet')