# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rluSIBqb9bqMPCszGOJjySW61Rsyife3
"""

!pip install transformers
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-cased')

import numpy as np

!pip install datasets

import torch
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets

data_set = load_dataset("nlu_evaluation_data",split='train')

data_alarm_audio= data_set.filter(lambda example: example['scenario'] == 'alarm' or example['scenario'] == 'audio')

from transformers import BertModel, BertTokenizer
model_name = 'bert-base-uncased'
labels = ["alarm_query","alarm_remove","alarm_set","audio_volume_down","audio_volume_mute","audio_volume_other","audio_volume_up"]

def createEmbeddings(input):


  tokenizer = BertTokenizer.from_pretrained(model_name)
  # load
  embedding_rep = {}
  model = BertModel.from_pretrained(model_name)
  for i in input :
      input_text = i
      input_ids = tokenizer.encode(input_text, add_special_tokens=True)
      input_ids = torch.tensor([input_ids])
      embedding_rep[i] = model(input_ids).last_hidden_state
  return embedding_rep

labels = ["alarm_query","alarm_remove","alarm_set","audio_volume_down","audio_volume_mute","audio_volume_other","audio_volume_up"]

label_embeddings = createEmbeddings(labels)

label_embeddings["alarm_remove"].shape

input_text=[]
for entry in data_alarm_audio:
  input_text.append(entry["text"])

input_text[0]

sentence_embeddings = createEmbeddings(input_text)

sentence_embeddings["wake me up at five am this week"].shape

def doCross(label_embedd,sentence_embedd):
  return np.dot(label_embedd,sentence_embedd)

ex1 = sentence_embeddings["wake me up at five am this week"][:, 0:1,:].detach().numpy()
ex2_1 = label_embeddings["alarm_query"][:, 0:1,:].detach().numpy()
ex2_2 = label_embeddings["alarm_set"][:, 0:1,:].detach().numpy()

print(doCross(np.squeeze(np.asarray(ex1)),np.squeeze(np.asarray(ex2_1))))

print(doCross(np.squeeze(np.asarray(ex1)),np.squeeze(np.asarray(ex2_2))))