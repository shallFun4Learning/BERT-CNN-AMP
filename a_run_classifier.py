
from asyncio import FastChildWatcher
import os
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from sklearn.metrics import confusion_matrix
# import evaluate
import datasets
from traitlets import Bool
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    BertConfig,
    #   BertPreTrainedModelConfig,
    AutoConfig,
    PretrainedConfig,
    AutoModel
)
import pandas as pd
import torch
import numpy as np
from torch import nn, scalar_tensor
from datasets import Dataset
from pathlib import Path, PurePath
from a_create_model import *
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,


)

import evaluate
# createFFdataset()
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool,default=False)
parser.add_argument('--eval', type=bool,default=False)
parser.add_argument('--k', type=int,default=1)
parser.add_argument('--dataname', type=str,default='')
# get dataset
dataname = ''  # 'Bi-LSTM'
k = -1
train_dataset_path = ''
test_dataset_path = ''
pretrained_model_path = ''

def compute_metrics(eval_preds, NUM_LABELS=2):
    scalar_name = ['accuracy',
                   'recall',
                   "matthews_correlation"]
    metric = evaluate.combine(scalar_name)
    logits, labels = eval_preds
    if NUM_LABELS == 1:
        logits01 = np.where(logits < 0.5, 0, 1)
    else:
        logits01 = np.argmax(logits, axis=-1).flatten()
    logits01 = logits01.astype(np.int16)
    labels = labels.astype(np.int16)
    labels = labels.flatten()

    tn, fp, fn, tp = confusion_matrix(y_pred=logits01,
                                      y_true=labels).ravel()

    specificity = tn/(fp+tn)
    res_dict = metric.compute(predictions=logits01, references=labels)
    res_dict['specificity'] = specificity
    return res_dict

outPutDir = Path(f'./finetune/{dataname}_{k}ker')
outPutDir.mkdir(parents=True, exist_ok=True)
bestModel = str(outPutDir)+os.sep+'bestModel'


def train(n0_static=True, eval=False):
    # getdataset
    train_dataset = Dataset.load_from_disk(train_dataset_path)
    train_dataset = train_dataset.with_format("torch")
    # get model
    model = FCBert(pretrained_model_path)
    model.setBertGrad(n0_static)
    model.freeze()
    # model.checkGrad()
    if eval:
        temp = train_dataset.train_test_split(test_size=0.1, shuffle=True)
        train_dataset = temp['train']
        dev_dataset = temp['test']
        # trainer args
        training_args = TrainingArguments(
            output_dir=str(outPutDir),
            num_train_epochs=50,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=0.1,
            # weight_decay=0.01,
            learning_rate=(2e-5),
            logging_dir=str(outPutDir)+os.sep+'logs',
            evaluation_strategy='epoch',  # 'no',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            logging_steps=50,
            # save_steps=100,
            save_strategy='epoch',
            save_total_limit=5,
            seed=980702,
            data_seed=980702,
            remove_unused_columns=True,
            lr_scheduler_type="cosine_with_restarts",
            optim='adafactor',  # 'adamw_torch',
            dataloader_drop_last=False
        )

        # trainer
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        )
    else:
        # trainer args
        training_args = TrainingArguments(
            output_dir=str(outPutDir),
            num_train_epochs=50,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=0.1,
            weight_decay=0.01,
            learning_rate=(2e-5),
            logging_dir=str(outPutDir)+os.sep+'logs',
            logging_steps=50,
            # save_steps=100,
            save_strategy='epoch',
            save_total_limit=2,
            seed=980702,
            data_seed=980702,
            remove_unused_columns=True,
            lr_scheduler_type="cosine_with_restarts",
            optim='adafactor',  # 'adamw_torch',
            dataloader_drop_last=False
        )

        # trainer
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        )
    trainer.train()
    trainer.save_model(bestModel)


# train()


def test():

    test_dataset = Dataset.load_from_disk(test_dataset_path)
    test_dataset = test_dataset.with_format("torch")
    model = FCBert(pretrained_model_path)
    model.load_state_dict(torch.load(bestModel+os.sep+'pytorch_model.bin'))
    model.eval()

    ff_outPutDir = bestModel
  
    test_args = TrainingArguments(
        output_dir=ff_outPutDir,
        per_device_eval_batch_size=32,
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=10,
        seed=980702,
        data_seed=980702,
        remove_unused_columns=True,
        dataloader_drop_last=False
    )

    trainer = Trainer(
        model=model,                         
        args=test_args,
        compute_metrics=compute_metrics,    
    )

    res = trainer.predict(test_dataset=test_dataset)
    print(res)
    print('test end.')


# test()

if __name__ == '__main__':
    args = parser.parse_args()
    k=args.k
    dataname=args.dataname
    train_dataset_path = f'./dataset/{str(k)}kmer_tfrecord/'+dataname+r'/trainhug'
    test_dataset_path = f'./dataset/{str(k)}kmer_tfrecord/'+dataname+r'/devhug'
    pretrained_model_path = f'./model/{str(k)}kmer_model'
    do_train=args.train
    do_eval=args.eval
    if do_train:
        train()
    if do_eval:
        test()
