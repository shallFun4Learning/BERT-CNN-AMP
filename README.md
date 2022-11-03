# BERT-CNN-AMP
We combine the pre-trained model BERT and Text-CNN to AMPs recognition.

## Requirements
cudatoolkit	10.1.243
cudnn	7.6.5
dataclasses	0.8
datasets	2.4.0
huggingface_hub	0.9.1
python	3.7.13
pytorch	1.7.1
readline	8.1.2
requests	2.28.1
scikit-learn	1.0.2
scipy	1.7.3
tensorboard	2.8.0
tensorflow-estimator	2.1.0
tensorflow-gpu	2.1.0
tokenizers	0.12.1
torchaudio	0.7.2
torchvision	0.8.2
tqdm	4.64.0
transformers	4.21.3

## Model&dataset

In order to test the effectiveness of this method, we compared it with the method of Zhang et al. under the same conditions. In this experiment, the data set and pre training BERT that are completely consistent with them are used.\
Pretaining datasets: click ->[Github Link](https://github.com/BioSequenceAnalysis/Bert-Protein)
\
Pretrained model: click ->[Onedrive Link](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EYnBCEVyN4FKjsSgAUgIDPMBQ6grhDA7_COYEOmu_FB5og?e=d9eaCQ)
\
Fine-tunning datasets (PyTorch available): click ->[Onedrive Link](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ES_Sj5aTpNlGvzETY700tIoBJgDrNk7AOBo-qadIfOyV3w?e=vkC0tY
)


## Usage

### tfrecord dataset 2 csv format
python tf2csv.py --k INT --dataname YOUR_DATA_NAME

### csv 2 datasets
python create_dataset.py -- test_path your/test_csv/path\
-- train_path your/train_csv/path

### Train&Test
python a_run_classifier.py --train True --eval True --k INT --dataname YOUR_DATA_NAME

(k means *k*mer_model)

