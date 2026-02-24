# Pipeline de Processamento de Imagem ECG — SLM MedTeacher

**Versão**: Fevereiro 2026
**Escopo**: Fluxo completo desde o upload da imagem até a resposta do modelo de linguagem.

---

## Visão Geral

Quando o usuário faz upload de uma imagem de ECG no modo `clinical-reasoning`, o sistema executa dois fluxos em paralelo que convergem antes de chegar ao modelo de linguagem (LLM):

1. **Fluxo de Percepção** — extração de dados técnicos do sinal (modelos de ML)
2. **Fluxo de Visão** — preparação da imagem bruta para envio inline ao LLM (base64)

Ambos alimentam um único prompt estruturado enviado ao modelo.

---

## Diagrama de Fluxo

```
[Upload: bytes da imagem]
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
   [FLUXO DE PERCEPÇÃO]                  [FLUXO DE VISÃO]
   ECGAnalyzer.analyze()              ImageProcessor.process_image()
         │                                          │
         ▼                                          ▼
   Pipeline ECG (3 níveis)         Redimensionamento → JPEG/PNG → base64
         │                                          │
         ▼                                          ▼
   get_summary_for_prompt()         image_base64_list[]
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
              [COMPOSIÇÃO DO PROMPT]
              perception_block + instruction_block + user_text
                        │
                        ▼
                   [LLM (Ollama / Gemini)]
                        │
                        ▼
                   [Resposta estruturada]
```

---

## Etapa 1 — Recepção e Classificação do Arquivo

**Componente**: `backend/main.py` → endpoint `POST /chat/multimodal`

O arquivo chega como `multipart/form-data`. O sistema:

1. Lê os bytes brutos e o filename original.
2. Determina o tipo de exame via função `_hint_from_file()`:
   - Verifica o filename: presença de palavras-chave como `ecg`, `ekg`, `eletro`, `electrocardiog`
   - Verifica o MIME type: `image/jpeg`, `image/png`, etc.
   - Fallback: usa o hint explícito enviado pelo frontend (`selected_exams`)
3. Normaliza o tipo: o mapa `EXAM_TYPE_MAP` converte strings do frontend (`rx`, `tc`, `lab`) para as chaves internas (`xray`, `ct`, `lab_results`).

> **Resultado**: cada arquivo recebe um `hint` normalizado — para o ECG, o valor será `"ecg"`.

---

## Etapa 2 — Pré-processamento de Imagem (Fluxo de Visão)

**Componente**: `backend/utils/image_processor.py` → `ImageProcessor.process_image()`

Executado via `MultimodalProcessor.process_files()`. Prepara a imagem para envio inline ao LLM.

### 2.1 Validação
- Tamanho máximo: **10 MB** (`MAX_FILE_SIZE_MB`)
- Formatos aceitos: `jpg`, `jpeg`, `png`, `gif`, `bmp`

### 2.2 Conversão de modo de cor
- Converte modos não-padrão (RGBA, P, CMYK) para **RGB** via Pillow.

### 2.3 Enhancement especial para ECG
Se `hint == "ecg"`, aplica `_enhance_for_ecg()`:
- **Contraste**: fator 1.8× (`ImageEnhance.Contrast`)
- **Nitidez**: fator 2.0× (`ImageEnhance.Sharpness`)

> Motivo: grades cinzas de ECG perdem legibilidade após compressão JPEG em resoluções baixas.

### 2.4 Redimensionamento
- Dimensão máxima: **512 px** no lado maior (`MAX_IMAGE_DIMENSION`)
- Algoritmo: Lanczos (melhor qualidade para downscaling)
- Proporção original preservada

### 2.5 Compressão e codificação
- PNG de entrada → salvo como **PNG** (sem perda)
- JPEG/outros → salvo como **JPEG** com qualidade **85%**
- Saída: string **base64** para envio inline ao LLM

> **Resultado**: imagem compactada em base64, formato e MIME type preservados para roteamento correto ao Gemini.

---

## Etapa 3 — Pipeline de Percepção ECG

**Componente**: `backend/utils/perception/ecg_analyzer.py` → `ECGAnalyzer.analyze()`

Este é o núcleo técnico do sistema. Executa em **3 níveis de degradação progressiva**, dependendo da disponibilidade dos modelos:

---

### Nível 1 — Pipeline Completo (Ahus-AIM disponível)

**Status de saída**: `success`

#### 3.1.a BHF Model (ConvNeXt — sempre executa)

- Modelo: `convnext_base.fb_in22k_ft_in1k` (timm, pré-treinado ImageNet)
- Input: imagem redimensionada para 224×224, normalização ImageNet
- Output: probabilidades sobre 6 classes: `['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']`
- Confiança marcada como `"low"` (cabeça classificadora sem fine-tuning cardíaco)

#### 3.1.b Digitização de Sinal — Ahus-AIM

- Ferramenta: `InferenceWrapper` (framework Ahus-AIM, carregado via `sys.path`)
- Pesos necessários (Git LFS):
  - `backend/models/ecg_segmentation_unet.pt` (segmentação de derivações)
  - `backend/models/lead_name_unet_weights_07072025.pt` (identificação de nomes de derivações)
- Antes de carregar, `_validate_weight_file()` verifica se o arquivo é ponteiro LFS (< 1024 bytes → erro)
- Layout restringido a `layout_should_include_substring="standard"` para evitar layout Cabrera (inverte aVR)
- Output: tensor `canonical_lines` — array NumPy de forma `(12, N)` (12 derivações, N amostras)

#### 3.1.c Queenbee (ECG Transformer)

- Modelo: `Trustcat/queenbee-ecg-transformer` (HuggingFace)
- Arquitetura: `ECGTransformer` (12 derivações, 5000 amostras, patch_size=50, embed_dim=256, 6 camadas, 8 heads)
- Input: sinal interpolado para exatamente 5000 amostras via `_prepare_signal()`
- Output (multi-label via sigmoid): probabilidades sobre 5 superclasses PTB-XL: `['NORM', 'MI', 'STTC', 'CD', 'HYP']`
- `weights_only=False` necessário (checkpoint usa `numpy._core.multiarray.scalar`)

#### 3.1.d HuBERT-ECG

- Modelo: `Edoardo-BS/hubert-ecg-base` (HuggingFace Foundation Model)
- Status atual: **retorna None** — o modelo base não possui cabeça classificadora fine-tuned. É incluído na arquitetura para futura integração após fine-tuning.

#### 3.1.e neurokit2 (Lead II)

- Extrai Lead II do sinal multi-derivação (`signal_np[0, 1, :]`)
- `nk.ecg_clean()` → `nk.ecg_peaks()` → intervalos R-R → FC
- Regularidade: RMSSD e coeficiente de variação (rr_cv > 0.20 ou RMSSD > 80 ms → "irregular")
- Duração do QRS via `nk.ecg_delineate()` (método CWT, best-effort)

#### 3.1.f Consenso Final

- Votos coletados de: BHF + Queenbee (HuBERT ignorado por retornar None)
- Empate resolvido por `CLINICAL_SEVERITY`: prioriza achado mais grave
  ```
  AF(6) > MI(6) > LBBB(5) > HYP(4) > RBBB(4) > CD(3) > 1dAVb(3) > STTC(2) > ST(2) > SB(1) > NORM(0)
  ```
- FC: preferencialmente calculada pelo neurokit2; fallback para `_estimate_hr()` via `scipy.find_peaks`

---

### Nível 2 — Pipeline de Recuperação (OpenCV + neurokit2)

**Condição de ativação**: Ahus-AIM indisponível (pesos LFS não materializados)
**Status de saída**: `partial_cv`

#### 3.2.a BHF Model

Executa normalmente (idêntico ao Nível 1).

#### 3.2.b Digitizador OpenCV — Extração da Tira de Ritmo

**Componente**: `backend/utils/perception/ecg_digitizer_cv.py` → `digitize_rhythm_strip()`

Extrai o sinal 1D da tira de ritmo (geralmente derivação II, faixa inferior do ECG):

| Etapa | Operação |
|---|---|
| 1 | Decodificação da imagem via `cv2.imdecode` |
| 2 | Conversão para escala de cinza |
| 3 | Recorte da região da tira de ritmo: **30% inferior** da imagem |
| 4 | **CLAHE** (Contrast Limited Adaptive Histogram Equalization): `clipLimit=2.0`, `tileGridSize=(8,8)` |
| 5 | Inversão adaptativa: se `mean(pixel) > 128` (fundo claro = papel ECG padrão) → inverte |
| 6 | **Threshold adaptativo gaussiano**: `blockSize=15`, `C=-5` → binariza o traço |
| 7 | **Extração por centro de massa por coluna**: para cada coluna x, calcula o centro de massa dos pixels ativos → obtém y(x) |
| 8 | Normalização: subtrai média, divide pelo desvio padrão |

- Estimativa de `sample_rate ≈ largura_da_tira / 10` (strip de 10 segundos padrão, mínimo 100 Hz)
- Rejeição automática se < 50% das colunas têm pixels ativos (traço muito fraco ou imagem ruidosa)
- Output: `{"signal": np.ndarray (1D, float32), "sample_rate": int, "method": "opencv_rule_based"}`

#### 3.2.c neurokit2 na Tira de Ritmo

Mesmo processamento do Nível 1, mas com o sinal 1D do OpenCV:
- `nk.ecg_clean()` → filtragem do sinal bruto
- `nk.ecg_peaks()` → detecção de picos R
- FC via intervalos R-R, ritmo (regular/irregular), QRS via delineamento CWT
- Threshold para fibrilação atrial/flutter: `rr_cv > 0.20` ou `RMSSD > 80 ms`

#### 3.2.d Resultado

FC e ritmo fornecidos pelo neurokit2 substituem a mensagem "FC não estimável". O resumo enviado ao LLM contém os dados calculados + instrução para completar os campos restantes visualmente.

---

### Nível 3 — Apenas BHF Visual

**Condição de ativação**: Ahus-AIM indisponível E OpenCV falha (imagem ilegível)
**Status de saída**: `partial`

Apenas o BHF ConvNeXt rodou. Resultado com confiança baixa; o LLM recebe instrução para preencher apenas 3 campos (Ritmo / FC estimada / Achado principal) a partir da imagem.

---

### Nível 4 — Erro Total

**Condição de ativação**: Todos os modelos falharam (inclusive BHF)
**Status de saída**: `error` (exceção no pipeline)

O LLM recebe a instrução de fallback visual completa.

---

## Etapa 4 — Geração do Resumo de Percepção

**Componente**: `ECGAnalyzer.get_summary_for_prompt()`

Converte o dict de resultados em texto estruturado para inclusão no prompt:

| Status | Template gerado |
|---|---|
| `success` | `[DADOS DE TELEMETRIA DO ECG]`: diagnóstico, FC, detalhes |
| `partial_cv` | `[DADOS ECG — OpenCV + neurokit2]`: FC/ritmo/QRS do neurokit2 + instrução para completar campos visuais |
| `partial` | `[DADOS PARCIAIS DE TELEMETRIA DO ECG]`: diagnóstico BHF + instrução 3-campos |
| `error` | `[ECG — ANÁLISE VISUAL NECESSÁRIA]`: fallback visual completo |

Threshold de confiança BHF: se `max_prob < 0.50`, substitui label por "Indeterminado (baixa confiança: X%)".

---

## Etapa 5 — Composição do Prompt Final

**Componente**: `backend/main.py` (bloco de composição)

O texto enviado ao LLM é montado nesta ordem:

```
[perception_block]       ← resumo do ECGAnalyzer (Etapa 4)
[instruction_block]      ← template do exame (exam_instructions.py)
[safe_full_text]         ← mensagem do usuário + PDFs (anonimizado)
```

### 5.1 Injeção do Template de Exame

Lógica de decisão em `main.py`:

- Se `ecg_had_partial_or_error == True` (tags "PARCIAIS", "PARTIAL", "ANÁLISE VISUAL NECESSÁRIA" detectadas no perception_block):
  - Injeta `ecg_visual_fallback`: instrui preenchimento de apenas 3 campos
- Caso contrário (status `success` ou `partial_cv`):
  - Injeta template ECG completo (`exam_instructions.py`):
    ```
    HD (achado crítico: FA, IAMCSST, BAVT...)
    FC / Ritmo / Eixo / Intervalos (PR, QRS, QTc) / Onda P / Complexo QRS / Onda T
    ```

### 5.2 Anonimização de PII

- `medical_anonymizer.anonymize()` aplicado ao texto do usuário + PDFs
- **Não** aplicado ao `perception_block` (gerado por máquina — Presidio causa falsos positivos)

### 5.3 Preflight de contexto (Ollama)

- Limite: `NUM_CTX_OLLAMA * 2 = 8192 caracteres`
- Se `user_message_for_model` ultrapassar esse limite em modo local (não Gemini), trunca com aviso

### 5.4 System Prompt

- Modo `clinical-reasoning`: `clinical-reasoning_{lang}.txt` (sem repetição)
- Modo `patient-communication`: idem com `apply_repetition=False` em `/chat/multimodal` (contexto curto)
  - Em `/chat` (texto puro): `apply_repetition=True` — dobra o prompt (Leviathan et al., 2025)

---

## Etapa 6 — Inferência no Modelo de Linguagem

**Componente**: `backend/main.py` → roteamento por tier

### 6.1 Local (Ollama — MedGemma 4B ou 27B)

- Endpoint: `POST http://localhost:11434/api/chat`
- Opções: `num_ctx=4096`, `temperature=0.1`, `repeat_penalty=1.5`, `num_predict=1024`
- Imagens anexadas como `messages[-1]["images"] = [base64, ...]`
- Timeout: 20 minutos

### 6.2 Cloud (Gemini 2.5 Flash)

- API: `generativelanguage.googleapis.com`
- Imagens como `inline_data` com MIME type real (preservado do image_processor)
- `maxOutputTokens=2048`, `temperature=0.7`
- System prompt injetado como turno user/model fictício no início do histórico

---

## Etapa 7 — Pós-processamento e Armazenamento

1. `filter_thinking_tokens()` — remove tokens `<think>...</think>` de modelos reasoning
2. `_sanitize_response()` — remove prefixos/sufixos de role (ex: "Assistente:", "AI:")
3. Resposta armazenada no histórico da sessão (JSON em disco)
4. Imagens **não** armazenadas no histórico (evitar crescimento excessivo); apenas texto descritivo do arquivo

---

## Resumo dos Modelos no Pipeline ECG

| Modelo | Tipo | Input | Output | Disponibilidade |
|---|---|---|---|---|
| ConvNeXt (BHF) | CNN (timm) | Imagem 224×224 | 6 classes (confiança baixa) | Sempre |
| Ahus-AIM | UNet + pipeline | Imagem → 12 derivações | Array (12, N) | Requer LFS |
| Queenbee | Transformer | Sinal 12-lead, 5000 amostras | 5 superclasses PTB-XL | Requer sinal |
| HuBERT-ECG | Foundation Model | Sinal (futuro) | None (sem head) | Integração futura |
| OpenCV digitizer | Rule-based | Imagem → tira de ritmo | Sinal 1D | Sempre |
| neurokit2 | Rule-based | Sinal 1D | FC, ritmo, QRS | Sempre (se instalado) |
| MedGemma / Gemini | LLM | Texto + imagem base64 | Laudo estruturado | Por tier |

---

## Dependências por Nível de Pipeline

```
Nível 1 (completo):
  torch, torchvision, timm, transformers, huggingface_hub
  yacs, scipy, neurokit2, cv2
  git lfs pull (pesos Ahus-AIM)

Nível 2 (OpenCV recovery):
  cv2, numpy, neurokit2
  (sem pesos externos)

Nível 3 (BHF visual only):
  torch, timm
  (sem pesos externos além do ImageNet pré-treinado)

Envio ao LLM:
  Pillow (image_processor)
  httpx (chamadas API)
  presidio-analyzer, presidio-anonymizer, spacy (anonimização)
```

---

## Notas de Limitação Clínica

- **BHF (ConvNeXt)**: cabeça classificadora treinada com pesos ImageNet genéricos — **não** foi fine-tuned em ECGs. Resultados com confiança marcada como baixa e threshold de aceitação em 50%.
- **OpenCV digitizer**: a estimativa de `sample_rate` depende da premissa de strip de 10 segundos. ECGs com velocidade não-padrão (50 mm/s) terão sample_rate subestimado em 2×.
- **neurokit2**: classificação de ritmo por rr_cv/RMSSD é um critério conservador, adequado para triagem mas não substitui análise clínica de FA vs flutter vs outras arritmias.
- **Queenbee**: modelo treinado em PTB-XL (ECGs de 12 derivações, 500 Hz). Sinal digitizado pelo OpenCV (1 derivação, sample_rate estimado) não é compatível com este modelo — por isso Queenbee só roda quando Ahus-AIM fornece sinal de 12 derivações.
- Todo output do pipeline é rotulado como **hipótese diagnóstica** no prompt. O LLM é instruído a não emitir diagnóstico definitivo e a incluir disclaimer de finalidade educacional.
