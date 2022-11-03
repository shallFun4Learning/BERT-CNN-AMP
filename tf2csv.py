import torch
import json
import tensorflow as tf
import numpy as np
from pathlib import *
import pandas as pd
from google.protobuf.json_format import MessageToJson


def addData(df,m):
    data={
        'input_ids':list(map(int,m['features']['feature']['input_ids']['int64List']['value'])),
        'attention_mask':list(map(int,m['features']['feature']['input_mask']['int64List']['value'])),
        'labels':list(map(int,m['features']['feature']['label_ids']['int64List']['value'])),
        'token_type_ids':list(map(int,m['features']['feature']['segment_ids']['int64List']['value'])),
    }
    # print(data)
    df=df.append(data,ignore_index=True)
    return df

def main(tfPath):
    for path in tfPath:
        if not path.exists():
            raise f'Not found this file:{path}.'
        df=pd.DataFrame(columns=['input_ids','attention_mask','token_type_ids','labels'])
        save_path=Path(str(path).replace('.tf_record','.csv'))
        save_path.parent.mkdir(parents=True,exist_ok=True)
        dataset = tf.data.TFRecordDataset(str(path))
        count=0
        is_realCount=0
        for d in dataset:
            ex = tf.train.Example()
            ex.ParseFromString(d.numpy())
            m = json.loads(MessageToJson(ex))
            df=addData(df,m)
            # print(m['features']['feature'].keys())
            if int(m['features']['feature']['label_ids']['int64List']['value'][-1])==1:
                is_realCount+=1
            count+=1
        df.to_csv(path_or_buf =str(save_path),index=None,sep='\t')
        print(f'{path.name}:count={count},is_realCount={is_realCount}')

import os
def checkPath(path):
    if not os.path.exists(path):
        raise Exception(f"Path error:{path}")
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int,default=1)
parser.add_argument('--dataname', type=str,default='')
    
if __name__ == '__main__':
    args = parser.parse_args()
    k=args.k
    dataname=args.dataname
    tfPath=[f'dataset/{str(k)}kmer_tfrecord/{dataname}/dev.tf_record',
            f'dataset/{str(k)}kmer_tfrecord/{dataname}/train.tf_record']
    for i in tfPath:
        checkPath(i)
    tfPath=[Path(i) for i in tfPath]
    main(tfPath)
    