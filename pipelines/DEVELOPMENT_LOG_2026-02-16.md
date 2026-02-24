# SLM MedTeacher - Diário de Desenvolvimento
**Data:** 16 de Fevereiro de 2026
**Objetivo:** Competição Kaggle-Google HAI-DEF (MedGemma Impact Challenge)

## 1. Mudanças de Escopo e Arquitetura
*   **Sepultamento do Modo A (Simulador):** Decidido remover o simulador acadêmico devido à baixa acurácia do modelo 4B em manter estados clínicos complexos.
*   **Foco no Modo B (Raciocínio Clínico) e Modo C (Comunicação):** Pivotagem para uma ferramenta de auxílio profissional "Privacy-First".
*   **Nova Arquitetura de 3 Tiers:**
    1.  **Local Standard:** MedGemma 1.5 4B (Privacidade total, leve).
    2.  **Local High-Perf:** MedGemma 27B (`thiagomoraes/medgemma-27b-it:Q4_K_M`) para hardware robusto.
    3.  **Hybrid Cloud:** MedGemma 4B para sanitização local + Gemini 2.5 Flash (Cloud) para inteligência máxima.

## 2. A "Batalha do ECG" - Problemas e Soluções
*   **Problema:** O modelo 4B falhou em identificar arritmias (FA) e infra/supra de ST apenas por visão.
*   **Causa:** Baixa resolução espacial da camada de visão e falta de escala (régua) no modelo.
*   **Solução (Pipeline Híbrido):** Implementado `backend/utils/ecg_digitizer.py`.
    *   Usa OpenCV para isolar o traçado.
    *   Detecta o pulso de calibração para definir escala (px/mV e px/ms).
    *   Mapeamento anatômico de 12 derivações (Layout 3x4 + 1).
    *   Extração matemática de "Ground Truth" (FC, Variabilidade R-R, desnível de ST em mV).
    *   Injeção desses dados brutos no prompt do MedGemma para ancorar o raciocínio.

## 3. Privacidade e Ética (Kaggle Ready)
*   **Anonimização de Prompt:** Injetadas instruções obrigatórias para detecção de PII em todos os prompts mestres.
*   **Mascaramento de Arquivos:** Nomes de arquivos são exibidos na UI e nos exports de forma anonimizada (ex: `Joã...va.pdf`).
*   **Sanitização Híbrida:** No modo Cloud, o 4B local serve como barreira de segurança antes do envio para a nuvem.

## 4. Debugging e Performance
*   **Segurança de API:** Implementado uso de arquivos `.env` (via `python-dotenv`) e `.gitignore` para proteger a `GEMINI_API_KEY`, impedindo sua exposição no código-fonte.
*   **Erro 504 (Timeout):** Resolvido reduzindo o contexto (`NUM_CTX`) para 8192, aumentando o timeout para 600s e limitando imagens a 768px.
*   **Dependências:** Instalado `opencv-python-headless` no venv e atualizado `requirements.txt`.
*   **Erros de Código:** Corrigidos problemas de indentação no `main.py` e `SyntaxError` no `ecg_digitizer.py`.

## 5. Próximos Passos
*   **Testar o Tier 27B:** Avaliar a velocidade do carregamento híbrido (12GB VRAM + RAM).
*   **Validar Tier Cloud:** Configurar `GEMINI_API_KEY` e testar o roteamento.
*   **Refinar PDF:** Melhorar a extração de dados estruturados em laudos laboratoriais que falharam hoje.
*   **Instalador para Leigos:** Pensar no empacotamento final (PyInstaller/One-click install).

---
*Log gerado automaticamente para referência na próxima sessão de desenvolvimento.*
