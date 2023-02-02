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

## Model&Dataset

In order to test the effectiveness of this method, we compared it with the method of Zhang et al. under the same conditions. In this experiment, the data set and pre training BERT that are completely consistent with them are used.\
Pretaining datasets: click ->[Github Link](https://github.com/BioSequenceAnalysis/Bert-Protein)
\
Pretrained model: click ->[Onedrive Link](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EYnBCEVyN4FKjsSgAUgIDPMBQ6grhDA7_COYEOmu_FB5og?e=d9eaCQ) 
\
Fine-tunning datasets (PyTorch available): click ->[Onedrive Link](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ES_Sj5aTpNlGvzETY700tIoBJgDrNk7AOBo-qadIfOyV3w?e=vkC0tY
) and [DRAMP](https://4wgz12-my.sharepoint.com/:f:/g/personal/admin_4wgz12_onmicrosoft_com/EvbcAEvuQoNEsDCmeapnRv0Bdv-oTe07zwp-hllC6YSeMA?e=qXZy9Y)  


## Usage

### tfrecord dataset 2 csv format
python tf2csv.py --k INT --dataname YOUR_DATA_NAME

### csv 2 datasets
python create_dataset.py -- test_path your/test_csv/path\
-- train_path your/train_csv/path

### Train&Test
python a_run_classifier.py --train True --eval True --k INT --dataname YOUR_DATA_NAME

(k means *k*mer_model)

# 补充材料

## 数据集来源

<table border="0" cellpadding="0" cellspacing="0" width="939" style="border-collapse: 
 collapse;table-layout:fixed;width:704pt">
 <colgroup><col width="189" style="mso-width-source:userset;width:141.75pt">
 <col width="375" span="2" style="mso-width-source:userset;width:281.25pt">
 </colgroup><tbody><tr height="30" style="mso-height-source:userset;height:22.5pt" id="r0">
<td height="28" class="x69" width="189" style="height:21pt;width:141.75pt;">数据集</td>
<td class="x69" width="375" style="width:281.25pt;">正样本来源</td>
<td class="x69" width="375" style="width:281.25pt;">负样本来源</td>
 </tr>
 <tr height="162" style="mso-height-source:userset;height:121.5pt" id="r1">
<td height="160" class="x70" style="height:120pt;">From AMPScan Vr.2</td>
<td class="x71">APD vr. 3数据库中按照以下标准筛选的具有抗菌活性的1778条抗菌肽序列。<br>1. 革兰氏阳性和/或革兰氏阴性细菌<br>2. CD-HIT 检测后过滤掉长度小于 10 个氨基酸或共享≥ 90% 序列同一性的样本</td>
<td class="x71">UniProt 数据库中按照下面两个条件标准筛选出1778条长度分布与抗菌肽序列接近的非抗菌肽序列。<br>1. 检索条件：更换"subcellular location" 为 "cytoplasm"<span style="mso-spacerun:yes;">&nbsp; </span>r删除与以下关键字匹配的条目 "antimicrobial", "antibiotic", "antiviral", "antifungal", "effector" or "excreted"<br>2. 筛选条件：去除序列长度小于 10 或样本序列同一性大于 40% 的样本</td>
 </tr>
 <tr height="118" style="mso-height-source:userset;height:88.5pt" id="r2">
<td height="116" class="x70" style="height:87pt;">From Bi-LSTM</td>
<td class="x71">抗菌活性肽结构数据库(DBAASP) 中按照下面条件筛选的2609条抗菌肽序列：<br>1. 至少对一种细菌具有抗菌活性<br>2. 长度超过 3 个且少于 55 个氨基酸<br>3. 不含任何非蛋白质或 D型氨基酸</td>
<td class="x71">UniProtKB数据库中根据亚细胞定位按照下面条件筛选的3170条非抗菌肽序列：<br>1. 已知功能<br>2. 未提及抗菌性<br>3. 无分泌肽序列</td>
 </tr>
 <tr height="162" style="mso-height-source:userset;height:121.5pt" id="r3">
<td height="160" class="x70" style="height:120pt;">From iAMP-2L</td>
<td class="x71">APD 数据库中按以下标准筛选不同功能类型的879个抗菌肽序列：<br>1. 包含“Antibacterial”, “Anticancer/tumor”, “Antifungal”, “Anti-HIV” and “Antiviral” 不同功能类型层次的。<br>2. 序列长度为5-100个氨基酸<br>3. 使用CD-HIT去除了40% 相似度的氨基酸冗余序列</td>
<td class="x71">UniProt数据库中根据以下条件过滤2405条非抗菌肽序列：<br>1. 带有“antimicrobial”、“antibiotic”、“fungicide”、“defensin”等特定注释的样本<br>2. 40%序列相似度</td>
 </tr>
 <tr height="206" style="mso-height-source:userset;height:154.5pt" id="r4">
<td height="204" class="x70" style="height:153pt;">From MAMPs-Pred</td>
<td class="x71">APD 数据库中按以下标准筛选抗菌肽序列：<br>1. 包含 匹配“Wound healing”, “Spermicidal”, “Insecticidal”, “Chemotactic”, “Antifungal”, “Anti-protist”, “Antioxidant”, “Antibacterial”, “Antibiotic”, “Antimalarial”, “Antiparasital”, “Antiviral”, “Anticancer/tumor”, “Anti-HIV”, “Proteinase inhibitor”<span style="mso-spacerun:yes;">&nbsp; </span>“Surface immobilized” 检索词的<br>2. 序列长度为5-100个氨基酸<br>3. 使用CD-HIT对大于180的样本进行去冗余处理</td>
<td class="x71">UniProt数据库中根据以下条件筛选10503个非抗菌肽序列：<br>1. 不含非天然氨基酸<br>2. 不在阳性样本中<br>3. 序列长度为5-100个氨基酸<br>除此之外， Pfam蛋白质家族数据库中获得了 109 个相同长度的非抗菌肽序列 </td>
 </tr>
 <tr height="118" style="mso-height-source:userset;height:88.5pt" id="r5">
<td height="116" class="x70" style="height:87pt;">From DRAMP</td>
<td class="x71">从DRAMP数据库中下载了“Antimicrobial_amps”数据：<br>1. 删除了长度超过上述四个数据集最大最小范围的肽段；<br>2. 删除了使用未知氨基酸的肽段<br>3. 删除了出现在负样本集中的肽段</td>
<td class="x71">未保证实验公平，负样本收集自上述四个数据集的负样本：<br>1. 删除了重复出现的样本<br>2. 删除了出现在正样本集中的样本</td>
 </tr>
<!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="189" style="width:141.75pt"></td>
  <td width="375" style="width:281.25pt"></td>
  <td width="375" style="width:281.25pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

## Dataset Source
<table border="0" cellpadding="0" cellspacing="0" width="939" style="border-collapse: 
 collapse;table-layout:fixed;width:704pt">
 <colgroup><col width="189" style="mso-width-source:userset;width:141.75pt">
 <col width="375" span="2" style="mso-width-source:userset;width:281.25pt">
 </colgroup><tbody><tr height="30" style="mso-height-source:userset;height:22.5pt" id="r0">
<td height="28" class="x69" width="189" style="height:21pt;width:141.75pt;">Dataset</td>
<td class="x69" width="375" style="width:281.25pt;">Source of positive samples</td>
<td class="x69" width="375" style="width:281.25pt;">Source of negative samples</td>
 </tr>
 <tr height="246" style="mso-height-source:userset;height:185pt" id="r1">
<td height="244" class="x70" style="height:183.5pt;">From AMPScan Vr.2</td>
<td class="x71">The 1778 AMP sequences screened from APD vr.3 database according to the following criteria.<br>1.Samples with antibacterial activity against Gram-positive and/or Gram-negative bacteria <br>2.Samples shorter than 10 amino acids in length or sharing ≥ 90% sequence identity after CD-HIT detection</td>
<td class="x71">From UniProt, 1778 non-AMP sequences with length distribution close to the AMP sequences were screened according to the following two conditional criteria.<br>1.Set "subcellular location" as cytoplasm and removed any entries that matched the following keywords: "antimicrobial", "antibiotic", "antiviral", "antifungal", "effector" or "excreted"<br>2.Removed the samples with sequence length less than 10 or samples sequence identity greater than 40%<br></td>
 </tr>
 <tr height="177" style="mso-height-source:userset;height:133pt" id="r2">
<td height="175" class="x70" style="height:131.5pt;">From Bi-LSTM</td>
<td class="x71">The 2609 AMP sequences screened from the Database of Antimicrobial Activity and Structure of Peptides (DBAASP) according to the following conditions.<br>1.Selected polypeptides with antibacterial activity against at least one bacterial species<br>2.Longer than 3 and less than 55 amino acids <br>3.Did not contain any non-proteinogenic or D-amino acids as positive samples</td>
<td class="x71">The 3170 non-AMP sequences were gathered from the UniProtKB according to the subcellular location parameter with the following criteria.<br>1.The same length<br>2.With known function<br>3.no mention of being antibacterial<br>3.not secreted peptide sequences </td>
 </tr>
 <tr height="162" style="mso-height-source:userset;height:121.5pt" id="r3">
<td height="160" class="x70" style="height:120pt;">From iAMP-2L</td>
<td class="x71">The 879 AMP sequences of different functional types<span style="mso-spacerun:yes;">&nbsp; </span>screened from APD database according to the following criteria.<br>1.Samples with sequence lengths ranging from 5 to 100 amino acids<br>2.Used CD-HIT to remove redundant sequences with 40% similarity</td>
<td class="x71">The 2405 non-AMP sequences came from UniProt. By filtering the samples with specific annotations such as "antimicrobial", "antibiotic", "fungicide", "defensin" and the samples with 40% sequence similarity,</td>
 </tr>
 <tr height="285" style="mso-height-source:userset;height:214pt" id="r4">
<td height="283" class="x70" style="height:212.5pt;">From MAMPs-Pred</td>
<td class="x71">The AMP sequences screened from APD database according to the following criteria. <br>1.Obtained “Wound healing”, “Spermicidal”, “Insecticidal”, “Chemotactic”, “Antifungal”, “Anti-protist”, “Antioxidant”, “Antibacterial”, “Antibiotic”, “Antimalarial”, “Antiparasital”, “Antiviral”, “Anticancer/tumor”, “Anti-HIV”, “Proteinase inhibitor” and “Surface immobilized” sequences <br>2. Samples with sequence lengths ranging from 5 to 100 amino acids <br>3.CD-HIT is used to perform redundancy removal to a subset of samples with sequence numbers larger than 180</td>
<td class="x71">The non-AMP sequences came from UniProt. By filtering the samples with the following criteria.<br>1.Did not contain unnatural amino acids<br>2.not in the positive samples<br>3.Between 5 and 100 amino acids in length.<br>In addition, they obtained 109 non-AMP sequences of the same length from the Pfam family.</td>
 </tr>
 <tr height="146" style="mso-height-source:userset;height:110pt" id="r5">
<td height="144" class="x70" style="height:108.5pt;">From DRAMP</td>
<td class="x71">Downloaded the "Antimicrobial_amps" data from the DRAMP database.In addition,removed petitdes whose length exceeded the length range of the four data sets mentioned above,with unknown amino acids and in negetive samples.</td>
<td class="x71">To ensure experimental fairness, negative samples were collected from the four data sets described above.<br>1. Samples with repeated occurrences were removed<br>2. The samples appearing in the positive sample set were removed</td>
 </tr>
<!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="189" style="width:141.75pt"></td>
  <td width="375" style="width:281.25pt"></td>
  <td width="375" style="width:281.25pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

## ROC图
（由于国内网络原因，可能出现无法显示的情况，下载后可正常查看）
![image](https://github.com/shallFun4Learning/BERT-CNN-AMP/blob/main/roc%E5%9B%BE_04.png)
