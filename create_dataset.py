import torch
import json
import tensorflow as tf
import numpy as np
from pathlib import *
import pandas as pd
from google.protobuf.json_format import MessageToJson
import torch
from datasets import Dataset
from ast import literal_eval
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,default='')
parser.add_argument('--train_path', type=str,default='')

def myconver(str):
    str = literal_eval(str)
    if not str:
        raise Exception(f'NULL!Please check:{str}')
    if not isinstance(str, list):
        raise Exception(f'Type error!Please check:{type(str)},{str}')
    return torch.tensor(str, dtype=torch.float64)


convert_dict = {
    'input_ids': literal_eval,
    'attention_mask': literal_eval,
    'labels': literal_eval,
    'token_type_ids': literal_eval,
}


def createDataset(df, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Dataset.from_pandas(df)
    save_path = path.replace('.csv', 'hug')
    data.save_to_disk(save_path)
    data = data.with_format("torch", device=device)
    print(save_path)


def main(train_csv, dev_csv):
    df_dev = pd.read_csv(dev_csv, sep='\t', converters=convert_dict)
    df_train = pd.read_csv(train_csv, sep='\t', converters=convert_dict)

    createDataset(df_dev, dev_csv)
    createDataset(df_train, train_csv)


def checkPath(path):
    if not os.path.exists(path):
        raise Exception(f"Path error:{path}")


if __name__ == '__main__':
    args = parser.parse_args()
    dev_csv = args.test_path
    train_csv = args.train_path
    checkPath(dev_csv)
    checkPath(train_csv)
    main(train_csv=train_csv, dev_csv=dev_csv)

