# SGERA
SGERA: Stein-Guided ECG-Report Alignment for ECG Representation Learning， ICML 2026

This is the code of SGERA: Stein-Guided ECG-Report Alignment for ECG Representation Learning. This paper is accepted by ICML 2026. The paper link will be released soon. 

## Abstract:
Electrocardiogram (ECG) representation learning via ECG-report alignment is often hindered by the inherent structural and statistical divergence between signals and natural language. Existing methods struggle to bridge this gap with simple contrastive objectives but struggle with distribution dependencies between heterogeneous features. To address this, we propose **SGERA** (**S**tein-**G**uided **E**CG-**R**eport **A**lignment), which leverages the unique properties of Stein kernels to provide a more rigorous geometric alignment in the latent space: **instance-level** alignment via a Stein-RBF kernel enforces pairwise consistency between ECG and report embeddings, and **distribution-level** alignment via a Stein-Score kernel captures higher-order interactions for global alignment. Furthermore, we introduce an ECG-Report matching task with a Hard Sample Mining strategy to refine discriminative boundaries. Experiments across three public datasets demonstrate that SGERA significantly outperforms state-of-the-art SSL methods in zero-shot classification, linear probing, and transfer learning, proving the superiority of Stein-guided alignment in handling complex medical modalities.


## Dataset
MIMIC-IV-ECG: We downloaded the MIMIC-IV-ECG dataset as the ECG signals and paired ECG reports: https://physionet.org/content/mimic-iv-ecg/1.0/

PTB-XL: We downloaded the PTB-XL dataset which consisting four subsets, Superclass, Subclass, Form, Rhythm: https://physionet.org/content/ptb-xl/1.0.3/

CPSC2018: We downloaded the CPSC2018 dataset which consisting three training sets: http://2018.icbeb.org/Challenge.html

CSN(Chapman-Shaoxing-Ningbo): We downloaded the CSN dataset: https://physionet.org/content/ecg-arrhythmia/1.0.0/


## Data Processing
We preprocessed pretraining datasets and split the dataset into train/val set using the code in pretrain/preprocess.ipynb.
We preprocessed downstream datasets and split the dataset into train/val/test set using the code in finetune/preprocess.ipynb.
We also provide the train/val/test split csv file in finetune/data_split

## Pre-training
bash MERL/pretrain/launch.sh

## Finetuning

**For zeroshot:**

cd MERL/zeroshot
bash zeroshot.sh

**For Linear probing:**
cd MERL/finetune/sub_script
bash run_all_linear.sh

## Acknowledgement
Thanks for the work of [MERL](https://github.com/cheliu-computation/MERL-ICML2024/tree/main). and [DERI](https://github.com/cccccj-03/DERI).
