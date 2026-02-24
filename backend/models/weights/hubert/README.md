---
license: cc-by-nc-4.0
library_name: transformers
tags:
- medical
---
# HuBERT-ECG: A Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

Original code at https://github.com/Edoar-do/HuBERT-ECG

License: CC BY-NC 4.0


## Abstract
Deep learning models have shown remarkable performance in electrocardiogram (ECG) analysis, but the limited availability and size of ECG datasets have constrained their success, resulting in systems that are more task specialists than versatile generalists. To counter this, we introduce HuBERT-ECG, a novel self-supervised foundation ECG model pre-trained on a large and diverse dataset of 9.1 million 12-lead ECGs encompassing 164 cardiovascular conditions. By simply adding a proper output layer, HuBERT-ECG can be fine-tuned for a wide array of downstream tasks, from diagnosing diseases to predicting future cardiovascular events. Across diverse real-world scenarios, HuBERT-ECG achieves AUROCs from 0.843 on small datasets to 0.99 on larger sources. When fine-tuned to detect 164 overlapping conditions simultaneously, our model delivers AUROCs above 0.9 and 0.95 for up to 140 and 97 diseases, respectively. HuBERT-ECG can also predict death events within a 2-year follow-up with AUROCs up to 0.91. We release pre-trained models and code as building baselines.

## Models
This repository contains the self-supervised pre-trained hubert-ecg-base

## Code

```python
from transformers import AutoModel
size = 'small' # any size from small, base, large
hubert_ecg = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{size}", trust_remote_code=True)
```

## IMPORTANT NOTE
Don't forget to pre-process your data! Read the paper to know more about it

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
https://doi.org/10.1101/2024.11.14.24317328
```