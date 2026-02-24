# SLM MedTeacher — Pipeline Completo
**Versão:** Fevereiro 2026
**Escopo:** Fluxo de dados, pesos, verificações de integridade e status de cada componente.

---

## 1. Visão Geral da Arquitetura

```
Usuário (Browser)
       │  HTTP/REST
       ▼
FastAPI Backend (backend/main.py)
       │
       ├──────────────────────────────────────────────────┐
       │                                                  │
       ▼                                                  ▼
Roteamento por Tier                              Camada de Percepção
  local_4b  → Ollama (MedGemma 4B)              PerceptionManager
  local_27b → Ollama (MedGemma 27B)                    │
  llm_cloud → OpenRouter API                    ┌───────┼──────────┐
                                                ▼       ▼          ▼
                                             ECG     X-Ray       OCR
                                          Analyzer  Analyzer  Analyzer
```

---

## 2. Tiers de Inteligência

| Tier ID       | Modelo                                       | Backend         | Requisito         |
|---------------|----------------------------------------------|-----------------|-------------------|
| `local_4b`    | `thiagomoraes/medgemma-1.5-4b-it:Q4_K_m`    | Ollama (local)  | ~3 GB VRAM        |
| `local_27b`   | `thiagomoraes/medgemma-27b-it:Q4_K_m`       | Ollama (local)  | ~16 GB VRAM       |
| `llm_cloud`   | Gemini 2.5 Flash / Kimi K2 / GLM-5           | OpenRouter API  | Chave em `~/.medteacher/vault.enc` |

**Lógica de fallback:** se `llm_cloud` selecionado mas sem chave API → downgrade automático para `local_4b`.

**Bug corrigido (2026-02-18):** comparação de model IDs era case-sensitive (`Q4_K_M` vs `Q4_K_m`), causando `tier_readiness = false` mesmo com modelos instalados.

---

## 3. Ciclo de Vida de uma Requisição

### 3.1 Texto puro (`POST /chat`)

```
[Usuário envia mensagem]
        │
        ▼
medical_anonymizer.anonymize()          ← Presidio + SpaCy: remove PII (PERSON, DATE, LOC)
        │
        ▼
detect_clinical_intent()                ← Classifica intent (presenting symptoms, exam result, etc.)
inject_intent_tag()                     ← Injeta tag [PRESENTING_SYMPTOMS] etc. no prompt
        │
        ▼
context_manager.should_compress()?
  → SIM: compress_context()             ← Mantém últimas 5 trocas; resume turnos antigos
  → NÃO: segue adiante
        │
        ▼
format_context_for_model()             ← system_prompt + consolidated_summary + recent_turns + user_msg
        │
        ├── Tier local → Ollama /api/chat (timeout 3 min)
        │       options: num_ctx=8192, temperature=0.1, repeat_penalty=1.5
        │
        └── Tier cloud → OpenRouter /api/v1/chat/completions (timeout 20 min)
                model_id: google/gemini-2.5-flash (ou config do LLM selecionado)
        │
        ▼
filter_thinking_tokens()               ← Remove tokens <unused94>…<unused95> do MedGemma
_sanitize_response()                   ← Hard cap 6000 chars + detecção de loop por n-grama
        │
        ▼
sessao.add_turn() → sessao.save()      ← Persiste em sessions/{id}/history.jsonl
        │
        ▼
[Resposta retornada ao cliente]
```

### 3.2 Com arquivos (`POST /chat/multimodal`)

```
[Usuário envia mensagem + arquivos]
        │
        ▼
_hint_from_file()                      ← MIME/nome → tipo de exame (ecg/xray/ct/lab_results)
        │
        ▼
MultimodalProcessor.process_files()
  ├── Imagem → ImageProcessor.process_image()
  │     → Redimensiona (≤1024px), boost de contraste para ECG, encode base64
  └── PDF   → PdfExtractor.extract_text()
        → PyPDF2: extrai texto, page count, word count
        │
        ▼
PerceptionManager.analyze_file()       ← Roteador de ML por tipo de exame
  ├── ECG       → ECGAnalyzer.analyze()        [ver seção 4]
  ├── X-Ray     → XRayAnalyzer.analyze()       [ver seção 5]
  ├── Lab/OCR   → DocumentAnalyzer.analyze()   [ver seção 6]
  ├── CT/MRI    → sem modelo ML; mensagem de aviso injetada no prompt
  └── PDF       → texto já extraído pelo PdfExtractor
        │
        ▼
EXAM_SPECIFIC_PROMPTS injection         ← prompts/exam_instructions.py
  → Formato estruturado por tipo de exame (ecg / rx / tc / lab)
  → Variante _llm para tiers cloud (prompt mais detalhado)
  → Se ECG teve erro parcial → usa ecg_visual_fallback (análise visual direta)
        │
        ▼
medical_anonymizer.anonymize(user_text) ← NÃO anonimiza perception_block (falsos positivos)
        │
        ▼
[Igual ao fluxo texto: context → LLM → sanitize → save]
```

---

## 4. Pipeline ECG — Detalhado

### 4.1 Visão Geral

```
[bytes da imagem ECG]
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
FLUXO DE PERCEPÇÃO              FLUXO DE VISÃO
ECGAnalyzer.analyze()           ImageProcessor
        │                       → base64 para LLM
        ▼
Etapa A: Digitalização (Imagem → Sinal)
        │
        ├─ Nível 1: SOTA (nnU-Net + Hough)    [ATIVO — M3/fold_all materializado]
        │
        └─ Nível 2: Ahus-AIM (U-Net fallback)  [ATIVO]
                → InferenceWrapper → sinal 12 derivações
        │
        ▼
Etapa B: Classificação (Sinal → Diagnóstico)
        │
        ├─ HuBERT ECG           [ATIVO]
        ├─ BHF ConvNeXt         [ATIVO — confiança baixa, pesos ImageNet]
        ├─ Queenbee Transformer  [ATIVO]
        ├─ CODE-15 ResNet        [ATIVO]
        └─ ECGFounder PKU        [ATIVO]
        │
        ▼
Consenso por voto ponderado (CLINICAL_SEVERITY)
        │
        ▼
get_summary_for_prompt()        → Bloco de texto estruturado injetado no prompt
```

### 4.2 Tabela de Pesos ECG

| Componente | Arquivo | Tamanho | SHA256 (primeiros 16 hex) | Status |
|------------|---------|---------|---------------------------|--------|
| Ahus-AIM Segmentation U-Net | `backend/models/ecg_segmentation_unet.pt` | 86.3 MB | `17fe7071ef270102...` | ✅ Verificado (LFS OID match) |
| Ahus-AIM Lead Identifier U-Net | `backend/models/lead_name_unet_weights_07072025.pt` | 22.2 MB | `840bd6bf2433ee6c...` | ✅ Verificado (LFS OID match) |
| CODE-15 ResNet (USP/UFMG) | `backend/models/resnet_code15.h5` | 24.6 MB | `93232c0dcecf2ac6...` | ✅ HDF5 válido (Zenodo/Dropbox oficial) |
| HuBERT ECG Foundation | `backend/models/weights/hubert/model.safetensors` | 355.3 MB | `05bc1b1317f8e306...` | ✅ Verificado (HF metadata match) |
| Queenbee Transformer | `backend/models/weights/queenbee/ecg_transformer_best.pt` | 56.3 MB | `bbf073249c9dca36...` | ✅ Verificado (HF metadata match) |
| ECGFounder (PKU) | HuggingFace Hub: `PKUDigitalHealth/ECGFounder` | ~600 MB | via HF Hub | ✅ Baixado sob demanda |
| BHF ConvNeXt | `timm`: `convnext_base.fb_in22k_ft_in1k` | ~350 MB | via timm/HF | ⚠️ Pesos ImageNet — sem fine-tuning ECG |
| nnU-Net SOTA (M3 fold_all best) | `weights/nnunet_ecg_sota/repo/models/M3/.../checkpoint_best.pth` | 453 MB | GitHub CDN (`media.githubusercontent.com`) | ✅ SHA256 `60020c47…` verificado |
| nnU-Net SOTA (M3 fold_all final) | `weights/nnunet_ecg_sota/repo/models/M3/.../checkpoint_final.pth` | 453 MB | GitHub CDN | ✅ SHA256 `8e4bae0b…` verificado |
| nnU-Net SOTA (M1 fold_0 final) | `weights/nnunet_ecg_sota/repo/models/M1/.../checkpoint_final.pth` | 453 MB | GitHub CDN | ✅ SHA256 `d0e0d4c5…` verificado |

### 4.3 Detalhes dos Modelos de Classificação

#### CODE-15 (ResNet — `resnet_code15.h5`)
- **Paper:** Ribeiro et al., *Nature Communications* 2020
- **Dataset:** 345 779 pacientes brasileiros (Clínicas UFMG)
- **Input:** `(N, 4096, 12)` — 4096 amostras @ 400 Hz, 12 derivações em µV×1e-4
- **Output:** `(N, 6)` — probabilidades para: 1dAVb, RBBB, LBBB, SB, AF, ST
- **Fonte:** `https://www.dropbox.com/s/5ar6j8u9v9a0rmh/model.zip?dl=0` (mirror oficial)
- **TensorFlow:** 2.20.0 instalado no venv

#### HuBERT ECG (`Edoardo-BS/hubert-ecg-base`)
- **Tipo:** Foundation model baseado em HuBERT (auto-supervised)
- **Input:** sinal bruto 500 Hz
- **Carregamento:** `AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True)`

#### Queenbee (`Trustcat/queenbee-ecg-transformer`)
- **Dataset:** PTB-XL
- **Classes:** 5 superclasses + 44 SCP codes
- **Input:** 12 derivações × 5000 amostras
- **Carregamento:** `snapshot_download` + instanciação manual via `model.py` do repo

#### ECGFounder (`PKUDigitalHealth/ECGFounder`)
- **Classes:** 150 diagnósticos diferentes
- **Arquitetura:** Net1D (MIT License, PKU)
- **Carregamento:** `snapshot_download` + `12_lead_ECGFounder.pth`

#### BHF ConvNeXt (modelo visual 2D)
- **Arquitetura:** `convnext_base.fb_in22k_ft_in1k` via `timm`
- **⚠️ Limitação:** cabeça classificadora usa pesos ImageNet, **não** fine-tuned para ECG. Resultados devem ter confiança baixa até fine-tuning ser realizado.

#### Ahus-AIM U-Net (Digitalizador Base)
- **Paper:** Stenhede et al., *npj Digital Medicine* 2026
- **Repo:** `Ahus-AIM/Open-ECG-Digitizer`
- **Arquitetura Segmentação:** U-Net `num_in=3, num_out=4, dims=[32,64,128,256,320,320,320,320], depth=2`
- **4 classes de saída:** grid (0), texto/fundo (1), sinal ECG (2), fundo (3)
- **Nota de carregamento:** pesos têm prefixo `_orig_mod.` (torch.compile); o `InferenceWrapper._load_segmentation_model_weights()` já remove esse prefixo automaticamente.

#### nnU-Net SOTA (Digitalizador Principal — FUNCIONAL)
- **Paper:** Krones, *PhysioNet Challenge 2024*
- **Repo:** `felixkrones/ECG-Digitiser` (cota LFS excedida no git; checkpoints baixados via GitHub CDN `media.githubusercontent.com`)
- **Modelo ativo:** M3/fold_all (3 checkpoints × ~453 MB, SHA256 verificados)
- **Pipeline implementado em** `ecg_digitiser_sota.py`:
  1. Decodificação da imagem (OpenCV)
  2. Detecção de rotação via Hough Transform (`_get_rotation_angle`)
  3. Segmentação via `nnUNetv2_predict` CLI (subprocess) com `nnUNet_results = M3/nnUNet_results`
  4. Máscara multi-classe → 12 máscaras binárias por derivação (`_cut_binary`)
  5. Calibração espacial (px/segundo, px/mV)
  6. Vectorização centroide por coluna → sinal 1D por derivação (`_vectorise`)
  7. Retorno: `np.ndarray (12, 5000)` em mV a 500 Hz, ordem `CANONICAL_LEAD_ORDER`
- **Dataset name corrigido:** `Dataset500_Signals` (era `Dataset123_ECG`)

---

## 5. Pipeline X-Ray (Radiografia de Tórax)

### 5.1 Fluxo

```
[bytes da imagem RX]
        │
        ▼
XRayAnalyzer.analyze()
        │
        ▼
Ensemble de 4 redes (carregamento lazy — primeiro uso):
  ├─ DenseNet121 (res224-all)        TorchXRayVision
  ├─ ResNet50    (res512-all)        TorchXRayVision
  ├─ DenseNet121 (NIH)               TorchXRayVision
  └─ DenseNet121 (CheXpert)          TorchXRayVision
        │
        ▼
Média ponderada de probabilidades por patologia
Threshold: _MIN_FINDING_PROB = 0.15
Top-K: _TOP_K = 5 achados reportados
        │
        ▼
get_summary_for_prompt()
  → Lista de achados em PT/EN com probabilidades
```

### 5.2 Pesos X-Ray

| Modelo | Fonte | Status |
|--------|-------|--------|
| DenseNet121 res224-all | TorchXRayVision (auto-download) | ✅ Baixado no primeiro uso |
| ResNet50 res512-all | TorchXRayVision (auto-download) | ✅ Baixado no primeiro uso |
| DenseNet121 NIH | TorchXRayVision (auto-download) | ✅ Baixado no primeiro uso |
| DenseNet121 CheXpert | TorchXRayVision (auto-download) | ✅ Baixado no primeiro uso |

**Nota:** `loaded: false` no `/health` é **comportamento normal** — lazy-loading; carrega no primeiro upload de RX.

---

## 6. Pipeline OCR (Laudos / Lab)

### 6.1 Fluxo

```
[bytes da imagem de laudo OU PDF]
        │
        ├─ PDF → PyPDF2.extract_text()          ← até 4000 chars do texto extraído
        │
        └─ Imagem → DocumentAnalyzer (PaddleOCR PP-OCRv4)
                → Detecção + reconhecimento de texto
                → get_summary_for_prompt()
```

### 6.2 Pesos OCR

| Modelo | Fonte | Status |
|--------|-------|--------|
| PaddleOCR PP-OCRv4 | PaddlePaddle Hub (auto-download) | ✅ Baixado no primeiro uso |

---

## 7. Pre-flight Check (`/health` e startup)

### 7.1 O que é verificado no startup

```python
# backend/main.py — lifespan()
1. CODE-15 (TensorFlow)     → TF instalado? arquivo .h5 > 0 bytes?
2. nnU-Net SOTA             → arquivos .pth existem E tamanho > 1 KB? (ponteiros LFS são 134 bytes)
3. HuBERT ECG               → model.safetensors existe?
4. Queenbee ECG             → ecg_transformer_best.pt existe?
5. Ahus-AIM                 → ecg_segmentation_unet.pt existe E tamanho > 1000 bytes?
```

### 7.2 Status atual do pre-flight

```
✅ CODE-15: Pesos detectados (24.6 MB)
✅ SOTA Digitiser: M3/fold_all materializado (3 checkpoints, 1359 MB total).
✅ HuBERT ECG: Pesos detectados (355.3 MB)
✅ Queenbee ECG: Pesos detectados (56.3 MB)
✅ Ahus-AIM: Pesos materializados
```

### 7.3 Endpoint `/health` — campos

```json
{
  "server": "healthy",
  "ollama": "connected",
  "models_in_ollama": ["thiagomoraes/medgemma-1.5-4b-it:Q4_K_m", "...27b..."],
  "tier_readiness": {
    "local_4b": true,
    "local_27b": true
  },
  "perception_readiness": {
    "ecg":  {"loaded": false/true, "error": null},
    "xray": {"loaded": false/true, "error": null},
    "ocr":  {"loaded": false/true, "error": null}
  }
}
```

`loaded: false, error: null` = lazy-loading; nunca foi invocado (normal).
`loaded: true` = instância criada e pronta.
`loaded: false, error: "..."` = falha na inicialização (problema real).

---

## 8. Segurança e Privacidade

### 8.1 Anonimização de PII

```
MedicalAnonymizer (Microsoft Presidio + SpaCy pt/en)
  Entidades redactadas: PERSON → [PACIENTE], DATE_TIME → [DATA], LOCATION → [LOCAL]
  Aplicado a: mensagem do usuário, texto de PDFs
  NÃO aplicado a: bloco de percepção (evita falsos positivos como "DIAGNÓSTICO" → "[PACIENTE]")
```

### 8.2 Cofre de API Keys (`Vault`)

```
~/.medteacher/vault.enc   ← ciphertext (AES-128-CBC + HMAC-SHA256, Fernet)
~/.medteacher/vault.key   ← chave simétrica (chmod 600)
Fallback: ~/.medteacher/vault.bin  ← Deceptive Vault legado (posições primas em ruído ASCII)
```

---

## 9. Gerenciamento de Sessões

```
Sessão = session_id (16 hex chars UUID4)
  sessions/{id}/metadata.json   ← modo, tier, idioma, turn_count, timestamps
  sessions/{id}/history.jsonl   ← uma linha JSON por mensagem (user + assistant)
  sessions/{id}/summary.txt     ← resumo consolidado (após compressão de contexto)

Limites:
  MAX_TURNS_PER_SESSION = 20
  SESSION_EXPIRY_HOURS  = 24
  MAX_CONTEXT_CHARS     = 128 000 (compressão automática a 80% = 102 400 chars)
  RECENT_TURNS_TO_KEEP  = 5 trocas completas

Cleanup:
  - Startup: remove sessões expiradas do disco
  - Periódico: a cada 30 minutos
  - Exports .md são preservados em sessions/exports/
```

---

## 10. Dependências Críticas

### 10.1 Python (venv)

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| fastapi | 0.115.0 | Servidor REST |
| uvicorn | 0.32.0 | ASGI server |
| httpx | 0.27.2 | Chamadas Ollama/OpenRouter |
| tensorflow | 2.20.0 | CODE-15 ResNet |
| torch | ≥2.1.0 | HuBERT, Queenbee, Ahus-AIM, X-Ray |
| torchxrayvision | latest | Ensemble X-Ray |
| timm | latest | BHF ConvNeXt |
| paddleocr | ≥3.4.0 | OCR de laudos |
| presidio-analyzer/anonymizer | latest | PII anonymization |
| spacy (pt_core_news_lg + en_core_web_lg) | latest | NER para Presidio |
| transformers | ≥4.40.0 | HuBERT loading |
| huggingface_hub | latest | Queenbee + ECGFounder download |
| cryptography | latest | Fernet vault |
| yacs | latest | Ahus-AIM config |

### 10.2 Externos

| Serviço | URL | Uso |
|---------|-----|-----|
| Ollama | localhost:11434 | Modelos locais MedGemma |
| OpenRouter | openrouter.ai/api/v1 | LLMs cloud |

---

## 11. Pendências e Limitações Conhecidas

| Item | Tipo | Detalhe |
|------|------|---------|
| BHF ConvNeXt | ⚠️ Baixa confiança | Pesos ImageNet, sem fine-tuning cardíaco. Resultado incluído no consenso mas com confiança declarada como `low`. |
| CT/MRI | ℹ️ Sem modelo | Sem CNN compatível disponível. LLM faz análise visual direta com instrução de formato específica. |
| nnU-Net checkpoints | ✅ Resolvido | Baixados via GitHub CDN (`media.githubusercontent.com`), SHA256 verificados — repo LFS excedido, CDN funciona como bypass. |
| SOTA Digitizer (`ecg_digitiser_sota.py`) | ✅ Resolvido | Implementado: Hough + nnU-Net subprocess + vectorização. Retorna `(12, 5000)` em mV. |
| Dataset name bug | ✅ Resolvido | `Dataset500_Signals` (era `Dataset123_ECG`). |
