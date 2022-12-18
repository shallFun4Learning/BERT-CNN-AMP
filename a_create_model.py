
import os
from typing import Optional
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from sklearn.metrics import confusion_matrix
# import evaluate
import datasets
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    BertModel,
    BertForPreTraining,
    BertForSequenceClassification,
    BertPreTrainedModel,
    PreTrainedModel)
import pandas as pd
import torch
import numpy as np
from torch import nn, scalar_tensor
from datasets import Dataset
from pathlib import Path, PurePath
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers import PretrainedConfig
from typing import List

'''
embeddings.word_embeddings.weight
embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight
embeddings.LayerNorm.bias

encoder.layer.0.attention.self.query.weight
encoder.layer.0.attention.self.query.bias
encoder.layer.0.attention.self.key.weight
encoder.layer.0.attention.self.key.bias
encoder.layer.0.attention.self.value.weight
encoder.layer.0.attention.self.value.bias
encoder.layer.0.attention.output.LayerNorm.weight
encoder.layer.0.attention.output.LayerNorm.bias
encoder.layer.0.intermediate.dense.weight
encoder.layer.0.intermediate.dense.bias
encoder.layer.0.output.dense.weight
encoder.layer.0.output.dense.bias
encoder.layer.0.output.LayerNorm.weight


pooler.dense.weight
pooler.dense.bias
'''


class FCBert(nn.Module):
    def __init__(self, path):
        super(FCBert, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.NUMLABELS = 2
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(1, 768), stride=1, padding=(
            0, 0), padding_mode='replicate', bias=True)
        self.conv2 = nn.Conv2d(1, 2, kernel_size=(3, 768), stride=1, padding=(
            1, 0), padding_mode='replicate', bias=True)
        self.conv3 = nn.Conv2d(1, 2, kernel_size=(5, 768), stride=1, padding=(
            2, 0), padding_mode='replicate', bias=True)
        # self.conv4=nn.Conv2d(1,2,kernel_size=(7,768),stride=1,padding=(3,0),padding_mode='replicate',bias=True)  
        self.mp = nn.MaxPool2d(kernel_size=(1, 2), stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(in_features=768, out_features=self.NUMLABELS)
        self.act = nn.Tanh()
        # self.lossfct=nn.CrossEntropyLoss()

    def setBertGrad(self, flag=True):
        for param in self.bert.parameters():
            param.requires_grad = flag
        print(f'Set bert.parameters {flag}.')

    def checkGrad(self):
        for param in self.bert.parameters():
            print(f'param={param},param.requires_grad={param.requires_grad}')

    def freeze(self):
        modules = [self.bert.embeddings,
                   ]  # self.albert.pooler self.bert.encoder,
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False
        #         print(param.requires_grad)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            print(param.requires_grad)
        # count = 0
        # for param in self.bert.encoder.parameters():
        #     if count < 6:
        #         param.requires_grad = False
        #         print(param.requires_grad)
        #     count += 1

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                # position_ids,
                labels):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        out = outputs['last_hidden_state']  # .permute(0,2,1)
        out = out.unsqueeze(dim=1)  # ([32, 128, 768])
        c1 = self.conv1(self.act(out))
        c2 = self.conv2(self.act(out))
        c3 = self.conv3(self.act(out))
        c = torch.cat((c1, c2, c3), dim=1).permute(
            0, 3, 2, 1)  # .reshape(out.shape[0],-1)
        c = (self.act(c)).reshape(out.shape[0], -1)
        logits = self.fc1(self.dropout(c))
        loss = None
        if self.NUMLABELS == 2:
            lossfct = nn.CrossEntropyLoss()
            
            '''
            #There is an application for Focal Loss. 
             lossfct = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                # alpha=torch.tensor([0.75, 0.25]),
                alpha=torch.tensor([0.52, 0.48]),
                gamma=2,
                reduction='mean',
                force_reload=False,
                verbose=False
            )
            lossfct.cuda()
            '''
            loss = lossfct(logits.view(-1, self.NUMLABELS), labels.view(-1))
        elif self.NUMLABELS == 1:
            labels = labels.float()
            # lossfct=nn.MSELoss()
            lossfct = nn.BCEWithLogitsLoss()
            loss = lossfct(logits.squeeze(), labels.squeeze())
        else:
            raise Exception(f'self.NUMLABELS wrong!')
        if loss is None:
            raise Exception(f'Loss type wrong!')
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
