# coding:utf-8
import modeling
import tokenization
from run_classifier import create_model, file_based_input_fn_builder
import tensorflow as tf
import optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
import os
import time

'''**********my work area**************'''

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
bestSP = -1
bestSE = -1
bestACC = -1
bestMCC = -1
bestAUC = -1

'''**********my work area**************'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean(
    'do_eval', True, 'Whether to evaluate after training')
tf.app.flags.DEFINE_boolean('do_save_model', True,
                            'Whether to save the model after training')
tf.app.flags.DEFINE_string('data_name', 'AMPScan',
                           "the name of the dataset to use")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('num_train_epochs', 50, 'training epochs')
tf.app.flags.DEFINE_float('warmup_proportion', 0.1, 'proportion of warmup')
tf.app.flags.DEFINE_float('learning_rate', 2e-5, 'learning rate')
tf.app.flags.DEFINE_boolean('using_tpu', False, 'Whether to use TPU')
tf.app.flags.DEFINE_float('seq_length', 128, 'Sequence length')
tf.app.flags.DEFINE_string(
    'data_root', './dataset/1kmer_tfrecord/AMPScan', 'The location of the data set to be used')
tf.app.flags.DEFINE_string(
    'vocab_file', './vocab/vocab_1kmer.txt', 'Dictionary location')
tf.app.flags.DEFINE_string(
    'init_checkpoint', "./model/1kmer_model/model.ckpt", 'Initialization node of the model')
tf.app.flags.DEFINE_string(
    'bert_config', "./bert_config_1.json", 'Bert configuration')
tf.app.flags.DEFINE_string(
    'save_path', "./model/AMPScan_1kmer_model/model.ckpt", 'Save location of pre-trained model')


def count_trues(pre_labels, true_labels):
    shape = true_labels.shape
    zeros = np.zeros(shape=shape)
    ones = np.ones(shape=shape)
    pos_example_index = (true_labels == ones)
    neg_example_index = (true_labels == zeros)
    right_example_index = (pre_labels == true_labels)
    true_pos_examples = np.sum(np.logical_and(
        pos_example_index, right_example_index))
    true_neg_examples = np.sum(np.logical_and(
        neg_example_index, right_example_index))
    return np.sum(pos_example_index), np.sum(neg_example_index), true_pos_examples, true_neg_examples


def main():
    '''**********my work area**************'''

    bestSP = -1
    bestSE = -1
    bestACC = -1
    bestMCC = -1
    bestAUC = -1

    '''**********my work area**************'''
    # The following are the input parameters.
    # When changing the dictionary, please modify the value of vocab_size in the file bert_config.json
    do_eval = FLAGS.do_eval
    do_save_model = FLAGS.do_save_model
    data_name = FLAGS.data_name
    # Record the number of samples in each data set
    train_dict = {"AMPScan": 2132,
                  "BiLSTM": 4174,
                  "iAMP": 1758,
                  "MAMPs": 5234,
                  "fold": 2928,
                  "all_data": 8978,
                  }
    test_dict = {"AMPScan": 1424,
                 "BiLSTM": 1156,
                 "iAMP": 1839,
                 "MAMPs": 1666,
                 "fold": 2119,
                 "all_data": 8978,
                 }
    tf.logging.set_verbosity(tf.logging.INFO)
    train_example_num = train_dict[data_name]
    test_example_num = test_dict[data_name]
    # If the GPU memory is not enough, you can consider reducing it
    batch_size = FLAGS.batch_size
    train_batch_num = math.ceil(train_example_num / batch_size)
    test_batch_num = math.ceil(
        test_example_num / batch_size)
    num_train_epochs = FLAGS.num_train_epochs
    warmup_proportion = FLAGS.warmup_proportion
    learning_rate = FLAGS.learning_rate
    # input(f'FLAGS.using_tpu:{FLAGS.using_tpu}')
    use_tpu = FLAGS.using_tpu
    # input(f'usetpu:{use_tpu}')
    seq_length = FLAGS.seq_length
    data_root = FLAGS.data_root
    vocab_file = FLAGS.vocab_file
    init_checkpoint = FLAGS.init_checkpoint
    # init_checkpoint =r'''./'''
    bert_config = modeling.BertConfig.from_json_file(
        FLAGS.bert_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Prevent directly occupying all GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    # Enter the training set, this file is generated using ljy_tsv2record
    input_file = data_root + "/train.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    num_train_steps = int(
        train_example_num / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    label_ids = tf.placeholder(
        dtype=tf.int32, shape=(None,))   # Leave four placeholders for entering data and labels
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    is_training = True
    num_labels = 2
    use_one_hot_embeddings = False
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        print(f'init_checkpoint:{init_checkpoint}')
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    drop_remainder = False

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    train_data = input_fn({"batch_size": batch_size})
    # Generate the training set data iterator, the iterator will output data in the loop
    iterator = train_data.make_one_shot_iterator().get_next()
    if do_eval:
        input_file = data_root + "/dev.tf_record"
        dev_data = input_fn({"batch_size": batch_size})
        dev_iterator = dev_data.make_one_shot_iterator().get_next()
    val_accs = []
    sps = []
    sns = []
    if do_save_model:
        saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(num_train_epochs):
            start_time = time.time()
            for _ in range(train_batch_num):
                # Run iterator to generate samples
                examples = sess.run(iterator)
                # print(examples)
                _, loss = \
                    sess.run([train_op, total_loss],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
            print("step:", step, " loss:", round(loss, 4), end=" ")
            all_prob = []
            all_labels = []
            all_pre_labels = []
            if not do_eval:
                end_time = time.time()
                eta_time = (end_time - start_time) * \
                    (num_train_epochs - step - 1)
                print(" eta time:", eta_time, "s")
                continue
            for _ in range(test_batch_num):
                examples = sess.run(dev_iterator)
                loss, prob = \
                    sess.run([total_loss, probabilities],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
                all_prob.extend(prob[:, 1].tolist())
                all_labels.extend(examples["label_ids"].tolist())
                pre_labels = np.argmax(prob, axis=-1).tolist()
                all_pre_labels.extend(pre_labels)
            acc = accuracy_score(all_labels, all_pre_labels)
            val_accs.append(acc)
            auc = roc_auc_score(all_labels, all_prob)
            mcc = matthews_corrcoef(all_labels, all_pre_labels)
            c_mat = confusion_matrix(all_labels, all_pre_labels)
            sn = c_mat[1, 1] / np.sum(c_mat[1, :])
            sp = c_mat[0, 0] / np.sum(c_mat[0, :])
            sps.append(sp)
            sns.append(sn)
            end_time = time.time()
            eta_time = (end_time - start_time) * (num_train_epochs - step - 1)
            print("SE:", sn, " SP:", sp, " ACC:", acc, " MCC:", mcc, " auROC:", auc, " eta time:",
                  eta_time, "s")
            # work
            bestSE = max(bestSE, sn)
            bestSP = max(bestSP, sp)
            bestACC = max(bestACC, acc)
            bestMCC = max(bestMCC, mcc)
            bestAUC = max(bestAUC, auc)
        if do_save_model:
            save_path = saver.save(
                sess, FLAGS.save_path)
            print('FLAGS.save_path={}', FLAGS.save_path)
            print('save_path={}', save_path)
        print("bestSE:", bestSE, " bestSP:", bestSP, " bestACC:",
              bestACC, " bestMCC:", bestMCC, " bestAUC:", bestAUC)


main()
