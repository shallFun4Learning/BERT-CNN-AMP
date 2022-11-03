export BERT_BASE_DIR=./model/1kmer_model

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/model.ckpt \
  --config ./bert_config_1.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin