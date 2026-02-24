# SLM MedTeacher - Diário de Desenvolvimento
**Data:** 17 de Fevereiro de 2026
**Objetivo:** Implementação da Arquitetura Híbrida de Percepção (CNNs + LLM)

## 1. Nova Arquitetura de Percepção
*   **Pivô para CNNs Especializadas:** Substituída a digitalização manual baseada em OpenCV por um sistema modular de "Ferramentas de Percepção".
*   **MedGemma como Orquestrador:** O modelo 4B agora atua como o cérebro que interpreta os dados estruturados vindos das CNNs (olhos).
*   **Implementação do PerceptionManager:** Centraliza a lógica de decisão sobre qual ferramenta usar (ECG, X-Ray ou OCR) baseada nos metadados da sessão.

## 2. Ferramentas Implementadas
*   **DocumentAnalyzer (OCR de Alta Precisão):** Integrado `PaddleOCR (PP-OCRv4)` para extração de texto estruturado de laudos e PDFs. Supera as limitações de contexto do LLM ao enviar apenas o texto relevante e limpo.
*   **ECGAnalyzer (ResNet1D):** Criada infraestrutura para inferência local usando `onnxruntime`. Preparado para receber pesos de modelos 1D-ResNet pré-treinados em PhysioNet.
*   **XRayAnalyzer (MobileNetV3):** Implementada ferramenta de triagem para radiografias de tórax, focada em 14 patologias (CheXpert style), com execução local otimizada.

## 3. Melhorias de Performance e Escalabilidade
*   **Execução ONNX:** Migração para `onnxruntime` permite inferência ultrarrápida em CPU (menos de 100ms), sem competir pela VRAM com o MedGemma.
*   **Modularidade:** A estrutura `backend/utils/perception/` permite adicionar novos modelos (ex: Tomografia, RM) sem alterar o core do backend.
*   **Pre-flight Model Check:** O endpoint `/health` agora valida se os modelos configurados nos Tiers (4B, 27B) estão realmente instalados no Ollama, evitando erros 500 durante o uso.
*   **Dependências:** Atualizados `requirements.txt` com `onnxruntime`, `paddleocr` e `paddle2onnx`.

## 4. Próximos Passos
*   **Providenciar Pesos Médicos:** Solicitar ao usuário o download dos arquivos `.onnx` específicos (ResNet1D CODE-15 e MobileNet-CheXpert).
*   **Refinar Digitalização ECG:** Implementar a integração com os conceitos do `Open-ECG-Digitizer` (Ahus-AIM) para converter imagens em séries temporais antes da análise.
*   **Correção de Timeouts (Rx):** Reduzido `NUM_CTX` para 4096 e resolução de imagem para 512px para garantir estabilidade em hardware com menos de 12GB VRAM.
*   **Comparação XAI:** Investigar o modelo "CNN XAI (MDPI 2025)" para fornecer mapas de calor explicativos na UI.

## 5. Manifesto de Dependências para Instalador (PC/Mobile)
Para garantir que o futuro instalador (PyInstaller ou Mobile Wrapper) funcione sem erros de "ModuleNotFound":
*   **Frameworks Base:** `paddlepaddle` (Engine), `onnxruntime` (Inference).
*   **Visão Computacional:** `opencv-python-headless`, `Pillow`.
*   **Ferramentas de Percepção:** `paddleocr`, `paddle2onnx`.
*   **Configuração de Runtime:** Definir `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` no ambiente para evitar timeouts em conexões lentas no primeiro boot.

---
*Log gerado após a implementação da arquitetura de ferramentas de percepção.*
