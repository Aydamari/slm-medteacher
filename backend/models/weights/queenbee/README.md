---
license: apache-2.0
tags:
- medical
- ecg
- cardiology
- transformer
- classification
- pytorch
- trustcat
- queenbee
datasets:
- ptb-xl
metrics:
- auc
- f1
pipeline_tag: other
language:
- en
library_name: pytorch
---

# 🐝 QueenBee ECG-Transformer

**Foundation model for 12-lead ECG diagnostic classification**

Part of the TrustCat sovereign medical AI stack - no cloud, no compromise.

## Model Description

A transformer-based architecture for multi-label ECG classification trained on PTB-XL, the largest freely accessible clinical ECG dataset (21,799 records).

### Architecture Highlights

- **Patch Embedding**: 100ms temporal patches (50 samples @ 500Hz)
- **Lead Embeddings**: Learnable embeddings for each of 12 leads
- **Rotary Position Embeddings**: Better temporal awareness than sinusoidal
- **Multi-head Self-Attention**: 8 heads × 6 layers
- **Multi-task Heads**: 5 superclasses + 44 SCP diagnostic codes

```
Input: [B, 12, 5000] (12-lead ECG, 10 seconds @ 500Hz)
  ↓
Patch Embedding (50 samples per patch)
  ↓
Lead Embeddings + Position Embeddings
  ↓
Transformer Encoder (6 layers, 8 heads, dim=256)
  ↓
[CLS] Token Pooling
  ↓
├── Superclass Head → [B, 5]  (NORM, MI, STTC, CD, HYP)
└── SCP Code Head   → [B, 44] (Detailed diagnostic codes)
```

## Performance

### Superclass Classification (5 classes)

| Class | Description | AUC | 
|-------|-------------|-----|
| NORM | Normal ECG | 92.8% |
| MI | Myocardial Infarction | 90.1% |
| STTC | ST/T Change | 91.6% |
| CD | Conduction Disturbance | 90.2% |
| HYP | Hypertrophy | 81.0% |
| **Mean** | **All Classes** | **89.1%** |

### Overall Metrics

| Metric | Value |
|--------|-------|
| Superclass Mean AUC | **89.1%** |
| Superclass F1 | 64.6% |
| SCP Code Mean AUC | 84.9% |

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | PTB-XL v1.0.3 |
| Train/Val/Test Split | 17,084 / 2,146 / 2,158 |
| Epochs | 50 (best @ epoch 10) |
| Optimizer | AdamW |
| Learning Rate | 1e-4 (cosine decay) |
| Batch Size | 64 |
| Hardware | 2× NVIDIA RTX 5090 |
| Training Time | ~18 minutes |

## Usage

```python
import torch
import wfdb
from scipy import signal as scipy_signal
from model import ECGTransformer

# Load model
model = ECGTransformer(
    num_leads=12,
    signal_length=5000,
    patch_size=50,
    embed_dim=256,
    depth=6,
    num_heads=8,
    num_superclasses=5,
    num_scp_codes=44
)

checkpoint = torch.load("ecg_transformer_best.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess ECG (example with WFDB)
signal, meta = wfdb.rdsamp("path/to/ecg_record")

# Resample to 500Hz if needed
if meta['fs'] != 500:
    signal = scipy_signal.resample(signal, int(len(signal) * 500 / meta['fs']), axis=0)

# Normalize per-lead
signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

# Pad/truncate to 5000 samples (10 seconds)
if len(signal) < 5000:
    signal = np.pad(signal, ((0, 5000 - len(signal)), (0, 0)))
else:
    signal = signal[:5000]

# Inference
x = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    superclass_logits, scp_logits = model(x)
    probs = torch.sigmoid(superclass_logits)[0]

classes = ["NORM", "MI", "STTC", "CD", "HYP"]
for cls, prob in zip(classes, probs):
    print(f"{cls}: {prob:.1%}")
```

## Superclass Definitions

| Code | Full Name | Clinical Significance |
|------|-----------|----------------------|
| NORM | Normal ECG | No significant abnormalities |
| MI | Myocardial Infarction | Heart attack (current or prior) |
| STTC | ST/T Change | Ischemia, electrolyte abnormalities |
| CD | Conduction Disturbance | Bundle branch blocks, AV blocks |
| HYP | Hypertrophy | Ventricular/atrial enlargement |

## Intended Use

- **Primary**: Clinical decision support for ECG interpretation
- **Secondary**: ECG screening and triage
- **Research**: Cardiology AI research and benchmarking

## Limitations

⚠️ **Not FDA Cleared** - For research use only

- Trained on PTB-XL dataset (German population, 1989-1996)
- HYP class has lower performance (smaller training set)
- Requires 12-lead ECG input (not suitable for single-lead devices)
- Should be validated on target population before clinical use

## Citation

If you use this model, please cite:

```bibtex
@misc{trustcat2026queenbee,
  title={QueenBee ECG-Transformer: Foundation Model for 12-Lead ECG Analysis},
  author={TrustCat},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/Trustcat/queenbee-ecg-transformer}
}
```

## License

Apache 2.0

---

**Built with diamond hands by TrustCat 🐝**

*Sovereign Medical AI - No Cloud, No Compromise*
