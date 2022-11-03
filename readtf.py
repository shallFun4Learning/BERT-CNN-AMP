import tensorflow as tf
import json
from google.protobuf.json_format import MessageToJson

import tensorflow as tf
path = 'dataset/1kmer_tfrecord/AMPScan/dev.tf_record'
# for example in tf.python_io.tf_record_iterator(path):
#     # print(tf.train.Example.FromString(example))
#     jsonMessage = MessageToJson(tf.train.Example.FromString(example))
dataset = tf.data.TFRecordDataset(path)
count=0
for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d.numpy())
    m = json.loads(MessageToJson(ex))
    print(m['features']['feature'].keys())
    count+=1
print(count)
print('end')
