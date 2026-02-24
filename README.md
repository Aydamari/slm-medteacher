# SLM MedTeacher

**Local AI Medical Teaching Assistant** — powered by [MedGemma 1.5 4B](https://developers.google.com/health-ai-developer-foundations/medgemma) (google/medgemma-1.5-4b-it), a Google Health AI Developer Foundations model.

> *Submission for the HAI-DEF Models Competition — February 2026*

---

## What it does

SLM MedTeacher is a privacy-first, offline-capable teaching assistant for medical students, interns, and residents. It combines a specialized medical perception pipeline with MedGemma's clinical reasoning to turn ECG images, chest X-rays, and lab results into structured teaching dialogues.

**Target user:** Medical interns and clinical-year students who need to understand ECGs and chest X-rays in real time — without internet, without cloud costs, and without sending patient data to third parties.

---

## Why MedGemma

MedGemma 1.5 4B is the only open, locally-deployable LLM trained on medical images and text. It runs on a laptop. It works offline. It doesn't cost per token and doesn't send patient data to external servers.

The 4B parameter size was a deliberate design choice with mobile deployment in mind. At Q4_K_M quantization, MedGemma 1.5 4B requires ~2.5 GB — within reach of 12 GB Android flagships (Samsung S24+, Pixel 9 Pro) and iPhone 15 Pro and newer, where llama.cpp already runs 7B models today via Apple's Neural Engine and CoreML. The specialized perception models (TorchXRayVision, ECG classifiers) are small enough for mobile conversion to ONNX or TFLite. A native smartphone app is the logical next step: the same architecture, the same privacy guarantee, delivered to the 6+ billion people carrying a capable computer in their pocket.

Our architecture uses MedGemma in two ways:
1. **LLM reasoning layer** — synthesizes structured perception outputs into clinical teaching dialogue
2. **Native vision** — directly interprets chest X-ray images via Ollama's vision API

This allows specialized models to handle signal extraction and pathology classification (what they do best), while MedGemma provides the medical reasoning and Socratic teaching layer (what it was trained for).

---

## Architecture

```
Upload (ECG / X-ray / Lab)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              PERCEPTION PIPELINE                     │
│                                                      │
│  ECG  → nnU-Net SOTA (PhysioNet 2024) digitizer     │
│          + CODE-15 + ECGFounder + Queenbee           │
│          + NeuroKit2                                 │
│                                                      │
│  X-Ray → TorchXRayVision (4-model ensemble)         │
│          + MedGemma 1.5 native vision  ← HAI-DEF    │
│                                                      │
│  Lab   → PaddleOCR + structured extraction          │
└──────────────────────┬──────────────────────────────┘
                       │  structured text block
                       ▼
        ┌──────────────────────────┐
        │    MedGemma 1.5 4B       │  ← HAI-DEF model
        │   (local via Ollama)     │
        │  Clinical reasoning +    │
        │  Socratic teaching       │
        └──────────────────────────┘
              or (cloud tier)
        ┌──────────────────────────┐
        │   Gemini 2.5 Flash       │
        │   via OpenRouter         │
        └──────────────────────────┘
```

**Privacy:** Microsoft Presidio + spaCy anonymization runs on all user input before it reaches any model. PERSON → `[PACIENTE]`, DATE → `[DATA]`, LOCATION → `[LOCAL]`.

---

## Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| **MedGemma 1.5 4B** | LLM reasoning, teaching, X-ray native vision | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| nnU-Net M3 | ECG signal segmentation (PhysioNet 2024 SOTA) | ECG-Digitiser / PhysioNet |
| CODE-15 ResNet | ECG classification (6 classes: AF, LBBB…) | CODE-15 project |
| ECGFounder Net1D | ECG classification (150 labels) | PKUDigitalHealth |
| Queenbee Transformer | ECG classification (MI, STTC, CD, HYP, NORM) | HuggingFace |
| NeuroKit2 | ECG: HR, HRV, QRS morphology, QTc | Rule-based |
| TorchXRayVision ×4 | Chest X-ray pathology detection | mlmed/torchxrayvision |
| PaddleOCR | Lab result text extraction | PaddlePaddle |
| Presidio + spaCy | PII anonymization | Microsoft / Explosion |

---

## Quick Start

**Requirements:** Python 3.10+, [Ollama](https://ollama.ai/download)

```bash
# Clone
git clone https://github.com/Aydamari/slm-medteacher
cd slm-medteacher

# Install (creates venv, installs deps, pulls MedGemma)
chmod +x setup.sh && ./setup.sh

# Start
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: **http://localhost:8000/medteacher.html**

For cloud LLMs (Gemini 2.5 Flash etc.), set your OpenRouter API key:
```bash
python installers/setup_secrets.py
```

**Model weights** (not included in repo — too large for GitHub):
See [`UPLOAD_GUIDE.md`](UPLOAD_GUIDE.md) for download instructions for the ECG ensemble weights.

---

## Modes

| Mode | Description |
|------|-------------|
| **Clinical Reasoning** | Uploads ECG/X-ray/lab → perception pipeline → MedGemma teaching dialogue |
| **Patient Communication** | Same pipeline, lay-language explanations |

Both modes support EN and PT (Portuguese).

---

## Impact

Brazil has ~400,000 enrolled medical students and ~50,000 active residents (CFM, 2024). Globally, the WHO estimates a shortage of 10 million healthcare workers by 2030, concentrated in settings where cloud AI is unavailable, unaffordable, or legally restricted for patient data.

SLM MedTeacher runs locally on a consumer laptop with no GPU required for the 4B model. No API key needed. No data leaves the device.

**Smartphone roadmap:** SLM MedTeacher is designed from the ground up to reach a smartphone. MedGemma 1.5 4B at Q4_K_M quantization (~2.5 GB) already runs on iPhone 15 Pro and newer via llama.cpp and Apple's Neural Engine, and on 12 GB Android flagships (Samsung S24+, Pixel 9 Pro). The perception models are small enough for ONNX/TFLite conversion. Porting the full stack to a native app would bring clinical AI teaching to the 6+ billion people carrying capable hardware in settings where cloud AI is unavailable, unaffordable, or legally restricted for patient data.

---

## Disclaimer

**SLM MedTeacher is an educational tool for medical students and healthcare professionals. It is not a medical device and is not intended for clinical diagnosis or treatment decisions. All AI outputs must be reviewed by a qualified and licensed healthcare professional before any clinical action is taken.**

---

## Author

Dr. Aydamari Faria Jr. — [github.com/Aydamari](https://github.com/Aydamari)

*Built for the HAI-DEF Models Competition, February 2026.*
