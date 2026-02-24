"""
FastAPI Backend - SLM MedTeacher
Servidor principal com endpoints REST
"""

import asyncio
import base64
import logging
import httpx
import json
import re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from backend.config import (
    PRODUCT_NAME,
    PRODUCT_VERSION,
    OLLAMA_BASE_URL,
    OLLAMA_API_TIMEOUT,
    MODELS_MAP,
    CORS_ORIGINS,
    SYSTEM_MESSAGES,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    MAX_TURNS_PER_SESSION,
    SESSIONS_DIR,
    NUM_CTX_OLLAMA,
    MODEL_TIERS,
    LLM_MODELS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from backend.models.sessao import GerenciadorSessoes
from backend.models.prompts import get_system_prompt
from backend.utils.context_manager import context_manager
from backend.utils.intent_detector import detect_clinical_intent, inject_intent_tag
from backend.models.multimodal import multimodal_processor
from backend.utils.perception.perception_manager import perception_manager, PerceptionManager
from backend.utils.security import medical_anonymizer
from prompts.exam_instructions import EXAM_SPECIFIC_PROMPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intervalo de limpeza de sessões expiradas (em segundos)
CLEANUP_INTERVAL_SECONDS = 30 * 60  # 30 minutos

gerenciador = GerenciadorSessoes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia startup e shutdown do app"""
    # Pre-flight check: Validar pesos e dependências críticas
    logger.info("--- Executando Checkup de Inicialização (Pre-flight) ---")
    
    # 1. Verificar CODE-15 (TensorFlow)
    try:
        import tensorflow as tf
        code15_path = Path("backend/models/resnet_code15.h5")
        if code15_path.exists():
            logger.info(f"✅ CODE-15: Pesos detectados ({code15_path.stat().st_size / 1024**2:.1f} MB)")
        else:
            logger.warning("❌ CODE-15: Arquivo 'resnet_code15.h5' não encontrado.")
    except ImportError:
        logger.warning("❌ CODE-15: TensorFlow não instalado no venv.")

    # 2. Verificar nnU-Net (SOTA 2024) — checar tamanho para descartar ponteiros LFS (134 bytes)
    nnunet_path = Path("backend/models/weights/nnunet_ecg_sota")
    m3_checkpoint = nnunet_path / "repo/models/M3/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth"
    real_pth_files = [p for p in nnunet_path.rglob("*.pth") if p.stat().st_size > 1_000_000]
    if m3_checkpoint.exists() and m3_checkpoint.stat().st_size > 1_000_000:
        total_mb = sum(p.stat().st_size for p in real_pth_files) / 1024**2
        logger.info(f"✅ SOTA Digitiser: M3/fold_all materializado ({len(real_pth_files)} checkpoints, {total_mb:.0f} MB total).")
    elif real_pth_files:
        logger.warning(f"⚠️  SOTA Digitiser: {len(real_pth_files)} checkpoint(s) detectados mas M3/fold_all incompleto.")
    else:
        lfs_count = sum(1 for p in nnunet_path.rglob("*.pth"))
        if lfs_count:
            logger.warning(f"❌ SOTA Digitiser: {lfs_count} checkpoint(s) são ponteiros LFS não materializados.")
        else:
            logger.warning("❌ SOTA Digitiser: Pesos nnU-Net não encontrados.")

    # 3. Verificar Ensemble de Sinais (ECGFounder + Queenbee)
    ecgfounder_net1d = Path("backend/models/weights/ecgfounder/net1d.py")
    ecgfounder_tasks = Path("backend/models/weights/ecgfounder/tasks.txt")
    queenbee_path    = Path("backend/models/weights/queenbee/ecg_transformer_best.pt")
    if ecgfounder_net1d.exists() and ecgfounder_tasks.exists():
        logger.info("✅ ECGFounder: Dependências locais detectadas (net1d.py + tasks.txt)")
    else:
        logger.warning("❌ ECGFounder: net1d.py ou tasks.txt ausentes em backend/models/weights/ecgfounder/")
    if queenbee_path.exists():
        logger.info(f"✅ Queenbee ECG: Pesos detectados ({queenbee_path.stat().st_size / 1024**2:.1f} MB)")

    # 4. Verificar Ahus-AIM (LFS check)
    aim_seg = Path("backend/models/ecg_segmentation_unet.pt")
    if aim_seg.exists() and aim_seg.stat().st_size > 1000:
        logger.info("✅ Ahus-AIM: Pesos materializados.")
    else:
        logger.error("❌ Ahus-AIM: Pesos são apenas ponteiros LFS. Execute 'git lfs pull'.")

    # Startup: limpeza inicial + agendar limpeza periódica
    gerenciador.cleanup_expired_from_disk()
    logger.info("Cleanup inicial de sessões expiradas concluído")

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    # Shutdown: cancelar task de limpeza
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


async def _periodic_cleanup():
    """Tarefa periódica que limpa sessões expiradas do disco"""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            gerenciador.cleanup_expired()
            gerenciador.cleanup_expired_from_disk()
        except Exception as e:
            logger.error(f"Erro no cleanup periódico: {e}")


app = FastAPI(
    title=PRODUCT_NAME,
    version=PRODUCT_VERSION,
    description="Local AI Medical Teaching Assistant",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class SessionCreateRequest(BaseModel):
    mode: str = "clinical-reasoning"
    language: str = DEFAULT_LANGUAGE
    model_tier: str = "local_4b"
    llm_model: str = "gemini_2.5_flash"
    exam_types: Optional[List[str]] = []


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = DEFAULT_LANGUAGE


def filter_thinking_tokens(text: str) -> str:
    """Remove thinking tokens do MedGemma (<unused94>thought...<unused95>)"""
    text = re.sub(r'<unused\d+>.*?<unused\d+>', '', text, flags=re.DOTALL)
    text = re.sub(r'<unused\d+>', '', text)
    return text.strip()


_MAX_RESPONSE_CHARS = 6000  # Limite máximo de caracteres por resposta

# E03: Orçamento de contexto para modelos Ollama (num_ctx = 4096 tokens)
# Reserva ~50% da janela para sistema + histórico + resposta
_OLLAMA_MAX_USER_MSG_CHARS = NUM_CTX_OLLAMA * 2  # 8192 chars ≈ 2048 tokens


def _hint_from_file(filename: str, content_type: str, fallback: str) -> str:
    """E05: Deriva exam type hint por MIME/nome de arquivo antes de cair no índice.

    Heurísticas sem ambiguidade: PDFs são laudos/lab; filenames com "ecg"/"ekg"
    são ECG. Para imagens genéricas (jpg/png sem nome sugestivo), usa o fallback
    fornecido pela posição no selected_exams.
    """
    fname = filename.lower()
    mime = content_type.lower()
    if mime == "application/pdf" or fname.endswith(".pdf"):
        return "lab_results"
    if any(kw in fname for kw in ("ecg", "ekg", "eletro", "electrocardiog")):
        return "ecg"
    if any(kw in fname for kw in ("rx", "xray", "radio", "chest", "torax", "tórax", "pulm")):
        return "xray"
    if any(kw in fname for kw in ("tc", "_ct", "tomografia", "scan", "abdome")):
        return "ct"
    return fallback


def _sanitize_response(text: str) -> str:
    """
    Detecta e trunca loops de geração antes de salvar no histórico.

    Estratégia:
    1. Hard cap de caracteres (evita respostas gigantes que envenenam contexto).
    2. Detecção de n-grama repetitivo: se uma sequência de >=40 chars se repete
       3+ vezes consecutivas, trunca antes da 2ª ocorrência.
    """
    if not text:
        return text

    # 1. Hard cap
    if len(text) > _MAX_RESPONSE_CHARS:
        text = text[:_MAX_RESPONSE_CHARS]
        logger.warning(f"Resposta truncada por hard cap ({_MAX_RESPONSE_CHARS} chars).")

    # 2. N-grama repetitivo — verificar substrings de 40-120 chars
    for ngram_len in (80, 40):
        for start in range(0, len(text) - ngram_len * 3, 10):
            chunk = text[start : start + ngram_len]
            pos2 = text.find(chunk, start + ngram_len)
            if pos2 == -1:
                continue
            pos3 = text.find(chunk, pos2 + ngram_len)
            if pos3 != -1:
                # Encontrou 3 ocorrências consecutivas do mesmo n-grama → truncar
                logger.warning(
                    f"Loop de geração detectado (ngram={ngram_len}). "
                    f"Truncando resposta em {pos2} chars."
                )
                return text[:pos2].rstrip()

    return text


async def check_models_locally() -> Dict[str, bool]:
    """Verifica quais modelos configurados estão realmente instalados no Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                installed = [m["name"] for m in response.json().get("models", [])]
                
                # Mapeia os IDs dos modelos configurados nos Tiers
                status = {}
                for tier_id, config in MODEL_TIERS.items():
                    model_id = config.get("model_id")
                    if config["type"] == "local" and model_id:
                        # Ollama as vezes oculta o :latest ou formata diferente
                        status[tier_id] = any(model_id.lower() in name.lower() for name in installed)
                return status
    except Exception:
        pass
    return {}


@app.get("/health")
async def health_check():
    """Verifica saúde do servidor, conectividade e presença de modelos"""
    # Ollama connectivity (isolated from ML dependencies)
    ollama_status = "disconnected"
    models_available = []
    ollama_error = None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"
        if response.status_code == 200:
            data = response.json()
            models_available = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        ollama_status = "error"
        ollama_error = str(e)

    # Tier model presence (uses Ollama)
    tier_readiness = {}
    try:
        tier_readiness = await check_models_locally()
    except Exception:
        pass

    # Perception ML dependencies (isolated — failure here ≠ Ollama failure)
    perception_readiness = {}
    perception_error = None
    try:
        perception_readiness = perception_manager.get_readiness()
    except Exception as e:
        perception_error = str(e)

    result = {
        "server": "healthy",
        "ollama": ollama_status,
        "models_in_ollama": models_available,
        "tier_readiness": tier_readiness,
        "perception_readiness": perception_readiness,
        "product": f"{PRODUCT_NAME} v{PRODUCT_VERSION}"
    }
    if ollama_error:
        result["ollama_error"] = ollama_error
    if perception_error:
        result["perception_error"] = perception_error
    return result


@app.post("/session/create")
async def create_session(request: SessionCreateRequest):
    """Cria nova sessão de estudo com suporte a Tiers"""
    try:
        # E08: Normalizar idioma inválido para o padrão
        if request.language not in SUPPORTED_LANGUAGES:
            logger.warning(f"[E08] Idioma '{request.language}' não suportado. Usando: {DEFAULT_LANGUAGE}")
            request.language = DEFAULT_LANGUAGE

        tier_config = MODEL_TIERS.get(request.model_tier, MODEL_TIERS["local_4b"])

        # Se for LLM cloud e não tiver chave, fallback para local
        if tier_config["type"] == "llm_cloud" and not OPENROUTER_API_KEY:
            logger.warning("Chave Gemini não encontrada. Fallback para Local 4B.")
            tier_config = MODEL_TIERS["local_4b"]
            request.model_tier = "local_4b"

        # Define modelo base: local usa Ollama; llm_cloud não usa Ollama
        if tier_config["type"] == "local":
            model = tier_config["model_id"]
        else:
            model = ""  # llm_cloud: nenhum modelo Ollama necessário

        # Validar llm_model se for tier cloud
        llm_model = request.llm_model
        if llm_model not in LLM_MODELS:
            llm_model = "gemini_2.5_flash"

        sessao = gerenciador.get_or_create(
            session_id=None,
            mode=request.mode,
            model=model,
            language=request.language,
            model_tier=request.model_tier,
            llm_model=llm_model
        )
        
        # Salvar tipos de exame nos metadados para uso posterior
        sessao.metadata["exam_types"] = request.exam_types
        
        sessao.save()
        
        logger.info(
            f"Sessão criada: {sessao.session_id} | "
            f"Modo: {request.mode} | "
            f"Idioma: {request.language}"
        )
        
        messages = SYSTEM_MESSAGES.get(request.language, SYSTEM_MESSAGES["en"])
        
        return {
            "session_id": sessao.session_id,
            "mode": request.mode,
            "language": request.language,
            "message": messages["session_created"],
            "created_at": sessao.created_at
        }
    
    except Exception as e:
        logger.error(f"Erro criando sessão: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Processa mensagem de chat"""
    try:
        sessao = gerenciador.get_or_create(session_id=request.session_id)

        # E04: Sanitizar respostas antigas ao carregar — evita reintrodução de loops em sessões salvas
        for _msg in sessao.conversation_history:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _msg["content"] = _sanitize_response(_msg["content"])

        if not sessao.can_continue():
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_TURNS_PER_SESSION} turns reached. Start new session."
            )
        
        is_first_turn = sessao.turn_count == 0
        is_clinical_reasoning = sessao.mode == "clinical-reasoning"
        needs_no_repetition = is_clinical_reasoning

        system_prompt = get_system_prompt(
            mode=sessao.mode,
            language=request.language,
            apply_repetition=not needs_no_repetition
        )
        
        if context_manager.should_compress(sessao.get_context_size_chars()):
            consolidated, recent = context_manager.compress_context(
                sessao.conversation_history,
                sessao.consolidated_summary
            )
            sessao.consolidated_summary = consolidated
            sessao.conversation_history = recent
        
        # Anonimização de PII (Privacy-First)
        safe_message = medical_anonymizer.anonymize(request.message, language=request.language)
        
        user_message_for_model = safe_message
        if is_clinical_reasoning:
            intent = detect_clinical_intent(safe_message)
            user_message_for_model = inject_intent_tag(safe_message, intent)
            logger.info(f"Clinical intent: {intent} for message: {safe_message[:60]}")

        messages = context_manager.format_context_for_model(
            system_prompt=system_prompt,
            consolidated_summary=sessao.consolidated_summary,
            recent_messages=sessao.get_recent_turns(5),
            new_user_input=user_message_for_model
        )
        
        try:
            # Lógica de Roteamento por Tier
            is_llm_cloud = sessao.model_tier in ("llm_cloud", "hybrid_cloud")
            if is_llm_cloud and OPENROUTER_API_KEY:
                # --- FLUXO LLM CLOUD via OpenRouter (OpenAI-compatible) ---
                llm_cfg = LLM_MODELS.get(sessao.llm_model, LLM_MODELS["gemini_2.5_flash"])
                logger.info(f"Roteando para OpenRouter: {llm_cfg['name']} ({llm_cfg['model_id']})")

                or_messages = []
                if system_prompt:
                    or_messages.append({"role": "system", "content": system_prompt})
                if sessao.consolidated_summary:
                    or_messages.append({"role": "user", "content": f"[Resumo do contexto anterior]\n{sessao.consolidated_summary}"})
                    or_messages.append({"role": "assistant", "content": "Contexto recebido."})
                for msg in sessao.get_recent_turns(5):
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    or_messages.append({"role": role, "content": msg["content"]})
                or_messages.append({"role": "user", "content": user_message_for_model})

                or_payload = {
                    "model": llm_cfg["model_id"],
                    "messages": or_messages,
                    "max_tokens": llm_cfg["max_output_tokens"],
                    "temperature": llm_cfg["temperature"],
                }
                or_headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://slm-medteacher.local",
                    "X-Title": "SLM MedTeacher",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(OPENROUTER_BASE_URL, headers=or_headers, json=or_payload, timeout=120)

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API Error: {response.text}")

                data = response.json()
                try:
                    assistant_response = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    assistant_response = "Error parsing OpenRouter response."

            else:
                # --- FLUXO LOCAL (OLLAMA) ---
                # Opções otimizadas para evitar loops e garantir resposta em 3 min
                ollama_payload = {
                    "model": sessao.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": NUM_CTX_OLLAMA,
                        "temperature": 0.1,      # Mais determinístico
                        "repeat_penalty": 1.5,   # Penalidade maior contra loops
                        # Stop apenas em padrões inequivocamente repetitivos.
                        # Removido "**HD (Hipótese Diagnóstica)**:" — cortava antes de gerar o diagnóstico.
                        # Removido "---" — é separador markdown legítimo em respostas estruturadas.
                        "stop": ["Usuário:", "\n\n\n\n"],
                        "num_predict": 1024      # Limita o tamanho da resposta
                    }
                }
                logger.info(f"Ollama call: model={sessao.model}, timeout=180s (3 min)")

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json=ollama_payload,
                        timeout=180.0  # Hard timeout de 3 minutos
                    )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama error: {response.text}"
                    )

                data = response.json()
                assistant_response = data.get("message", {}).get("content", "")
                assistant_response = filter_thinking_tokens(assistant_response)
                # Loop/hard-cap sanitization only needed for local models
                assistant_response = _sanitize_response(assistant_response)

            sessao.add_turn(safe_message, assistant_response)
            sessao.save()
            
            return {
                "response": assistant_response,
                "turn_count": sessao.turn_count,
                "context_usage": context_manager.get_context_usage_report(
                    sessao.get_context_size_chars()
                )
            }
        
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama request timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/multimodal")
async def chat_multimodal(
    session_id: str = Form(...),
    message: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    exam_types: Optional[str] = Form(None), # Recebe como string JSON
    files: List[UploadFile] = File(None)
):
    """Processa mensagem com arquivos anexados (imagens, PDFs)"""
    try:
        sessao = gerenciador.get_or_create(session_id=session_id)

        # E04: Sanitizar respostas antigas ao carregar — evita reintrodução de loops em sessões salvas
        for _msg in sessao.conversation_history:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _msg["content"] = _sanitize_response(_msg["content"])

        # Atualizar tipos de exame se enviados
        if exam_types:
            try:
                parsed_exams = json.loads(exam_types)
                sessao.metadata["exam_types"] = parsed_exams
                logger.info(f"Metadata updated for session {session_id}: {parsed_exams}")
            except Exception as e:
                logger.warning(f"Erro ao parsear exam_types dinâmicos: {e}")

        if not sessao.can_continue():
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_TURNS_PER_SESSION} turns reached. Start new session."
            )

        is_clinical_reasoning = sessao.mode == "clinical-reasoning"

        # Multimodal endpoint: always disable prompt repetition (num_ctx too tight)
        system_prompt = get_system_prompt(
            mode=sessao.mode,
            language=language,
            apply_repetition=False
        )

        # Process uploaded files
        file_data_list = []
        if files:
            for upload_file in files:
                if upload_file.filename:
                    content = await upload_file.read()
                    file_data_list.append({
                        "filename": upload_file.filename,
                        "content": content,
                        "content_type": upload_file.content_type or ""
                    })

        processed_data = None
        image_base64_list = []
        file_context_text = ""
        # #9: Anonymize user message unconditionally (mirrors /chat behaviour)
        user_text = medical_anonymizer.anonymize(message, language=language)

        if file_data_list:
            # #3: Normalise exam types BEFORE process_files so ECG images get contrast boost
            raw_exams = sessao.metadata.get("exam_types", [])
            selected_exams = [PerceptionManager.normalize_exam_type(e) for e in raw_exams] if raw_exams else ["general"]

            # E05: Build per-file exam_type_hints with MIME/name awareness
            file_exam_hints = [
                _hint_from_file(
                    file_data_list[i]["filename"],
                    file_data_list[i].get("content_type", ""),
                    selected_exams[i] if i < len(selected_exams) else selected_exams[0]
                )
                for i in range(len(file_data_list))
            ]

            processed_data = multimodal_processor.process_files(
                file_data_list, exam_type_hints=file_exam_hints
            )

            if processed_data.get("errors"):
                for err in processed_data["errors"]:
                    logger.warning(f"File processing error: {err['filename']}: {err['error']}")

            # Prepare text + images for model
            full_text, image_base64_list = multimodal_processor.prepare_for_model(
                processed_data, message
            )

            # #4: Per-file routing via MIME/extension + normalised hint
            perception_summaries = []
            ecg_had_partial_or_error = False

            for idx, f_data in enumerate(file_data_list):
                # E05: Use MIME/name-aware hint (file_exam_hints already computed above)
                hint = file_exam_hints[idx]

                summary = perception_manager.analyze_file(
                    f_data['content'],
                    f_data['filename'],
                    exam_type=hint,
                    content_type=f_data.get('content_type', ''),
                    language=language,
                )
                if summary:
                    perception_summaries.append(summary)
                    # E06: Detectar parcial pelo conteúdo do summary — sem depender do hint
                    # (tags são exclusivas do ecg_analyzer; falsos positivos em outros analyzers são impossíveis)
                    if any(tag in summary for tag in ["PARCIAIS", "PARTIAL", "ANÁLISE VISUAL NECESSÁRIA", "VISUAL ANALYSIS"]):
                        ecg_had_partial_or_error = True

            # #5: Inject exam-specific prompts, but skip rigid ECG prompt
            #     when ECG perception already returned partial/error
            # Mapa: tipo normalizado → chave base em EXAM_SPECIFIC_PROMPTS
            _NORM_TO_PROMPT_KEY = {"xray": "rx", "ct": "tc", "lab_results": "lab"}
            is_llm_tier = sessao.model_tier in ("llm_cloud", "hybrid_cloud")

            exam_instructions = []
            for exam in selected_exams:
                if exam == "ecg" and ecg_had_partial_or_error:
                    # Inject visual-fallback instruction instead of the rigid structured format
                    fb = EXAM_SPECIFIC_PROMPTS.get("ecg_visual_fallback", {})
                    fb_instr = fb.get(language, fb.get("en", ""))
                    if fb_instr:
                        exam_instructions.append(f"---\n{fb_instr}\n---")
                    continue  # Skip rigid format

                # Resolver chave base (xray→rx, ct→tc, etc.)
                base_key = _NORM_TO_PROMPT_KEY.get(exam, exam)

                # Para LLM, tentar variante _llm primeiro
                prompt_entry = None
                if is_llm_tier:
                    prompt_entry = EXAM_SPECIFIC_PROMPTS.get(f"{base_key}_llm")
                if prompt_entry is None:
                    prompt_entry = EXAM_SPECIFIC_PROMPTS.get(base_key)

                if prompt_entry:
                    instr = prompt_entry.get(language, prompt_entry.get("en", ""))
                    exam_instructions.append(f"---\n{instr}\n---")
            
            perception_block = "\n".join(perception_summaries)
            instruction_block = "\n".join(exam_instructions)

            # Anonimizar APENAS o texto de origem do usuário (full_text = user msg + PDFs).
            # O perception_block é gerado por máquina — anonimizá-lo causa falsos positivos
            # (ex: Presidio substituindo "DIAGNÓSTICO" por "[PACIENTE]"), corrompendo dados.
            safe_full_text = medical_anonymizer.anonymize(full_text, language=language)
            user_text = f"{perception_block}\n\n{instruction_block}\n\n{safe_full_text}"

            file_context_text = processed_data.get("context_summary", "")
        
        # Apply intent detection for clinical reasoning
        user_message_for_model = user_text
        has_files = bool(file_data_list)
        if is_clinical_reasoning:
            intent = detect_clinical_intent(message, has_files=has_files)
            user_message_for_model = inject_intent_tag(user_text, intent)
            logger.info(f"Clinical intent (multimodal): {intent}, files={len(file_data_list)}")

        # E03: Preflight de orçamento de contexto para Ollama (não se aplica ao Gemini)
        # Trunca user_message_for_model se exceder ~50% da janela num_ctx,
        # reservando espaço para system prompt + histórico + resposta.
        if sessao.model_tier not in ("llm_cloud", "hybrid_cloud") and len(user_message_for_model) > _OLLAMA_MAX_USER_MSG_CHARS:
            logger.warning(
                f"[E03] Mensagem multimodal ({len(user_message_for_model)} chars) excede "
                f"orçamento Ollama ({_OLLAMA_MAX_USER_MSG_CHARS} chars). Truncando."
            )
            user_message_for_model = (
                user_message_for_model[:_OLLAMA_MAX_USER_MSG_CHARS]
                + "\n[...conteúdo truncado por limite de contexto do modelo]"
            )

        # Compress context if needed
        if context_manager.should_compress(sessao.get_context_size_chars()):
            consolidated, recent = context_manager.compress_context(
                sessao.conversation_history,
                sessao.consolidated_summary
            )
            sessao.consolidated_summary = consolidated
            sessao.conversation_history = recent

        messages = context_manager.format_context_for_model(
            system_prompt=system_prompt,
            consolidated_summary=sessao.consolidated_summary,
            recent_messages=sessao.get_recent_turns(5),
            new_user_input=user_message_for_model
        )

        try:
            # Lógica de Roteamento por Tier (multimodal)
            is_llm_cloud = sessao.model_tier in ("llm_cloud", "hybrid_cloud")
            if is_llm_cloud and OPENROUTER_API_KEY:
                # --- FLUXO LLM CLOUD via OpenRouter (DIRETO — dados de percepção vão diretamente ao LLM) ---
                llm_cfg = LLM_MODELS.get(sessao.llm_model, LLM_MODELS["gemini_2.5_flash"])
                logger.info(f"Roteando multimodal para OpenRouter: {llm_cfg['name']} ({llm_cfg['model_id']})")

                or_messages = []
                if system_prompt:
                    or_messages.append({"role": "system", "content": system_prompt})
                if sessao.consolidated_summary:
                    or_messages.append({"role": "user", "content": f"[Resumo do contexto anterior]\n{sessao.consolidated_summary}"})
                    or_messages.append({"role": "assistant", "content": "Contexto recebido."})
                for msg in sessao.get_recent_turns(5):
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    or_messages.append({"role": role, "content": msg["content"]})

                # Mensagem atual: texto + imagens (somente se o modelo suporta visão)
                supports_vision = llm_cfg.get("vision", False)
                image_formats = []
                if processed_data:
                    for img in processed_data.get("images", []):
                        fmt = img.get("format", "jpeg").lower()
                        image_formats.append(f"image/{fmt}" if fmt != "jpg" else "image/jpeg")

                if supports_vision and image_base64_list:
                    current_content = [{"type": "text", "text": user_message_for_model}]
                    for i, img_b64 in enumerate(image_base64_list):
                        mime = image_formats[i] if i < len(image_formats) else "image/jpeg"
                        current_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                        })
                    or_messages.append({"role": "user", "content": current_content})
                else:
                    or_messages.append({"role": "user", "content": user_message_for_model})

                or_payload = {
                    "model": llm_cfg["model_id"],
                    "messages": or_messages,
                    "max_tokens": llm_cfg["max_output_tokens"],
                    "temperature": llm_cfg["temperature"],
                }
                or_headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://slm-medteacher.local",
                    "X-Title": "SLM MedTeacher",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(OPENROUTER_BASE_URL, headers=or_headers, json=or_payload, timeout=1200)

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API Error: {response.text}")

                data = response.json()
                try:
                    assistant_response = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    assistant_response = "Error parsing OpenRouter response."

            else:
                # --- FLUXO LOCAL (OLLAMA) ---
                ollama_payload = {
                    "model": sessao.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": NUM_CTX_OLLAMA,
                        "temperature": 0.1,
                        "repeat_penalty": 1.5,   # Previne loop de geração (crítico para ECG)
                        "stop": ["Usuário:", "\n\n\n\n"],
                        "num_predict": 1024
                    }
                }

                # Add images to Ollama payload if present
                if image_base64_list:
                    ollama_payload["messages"][-1]["images"] = image_base64_list

                logger.info(
                    f"Ollama multimodal call: model={sessao.model}, "
                    f"num_ctx={NUM_CTX_OLLAMA}, messages={len(messages)}, "
                    f"images={len(image_base64_list)}"
                )

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json=ollama_payload,
                        timeout=OLLAMA_API_TIMEOUT
                    )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama error: {response.text}"
                    )

                data = response.json()
                assistant_response = data.get("message", {}).get("content", "")
                assistant_response = filter_thinking_tokens(assistant_response)
                # Loop/hard-cap sanitization only needed for local models
                assistant_response = _sanitize_response(assistant_response)

            # Store in history: text only (no base64 images — too large)
            # Use anonymized message for privacy — raw `message` must not reach disk.
            history_user_text = medical_anonymizer.anonymize(message, language=language)
            if file_context_text:
                history_user_text = f"{message}\n\n{file_context_text}"

            # Extrair nomes reais dos arquivos para o histórico
            actual_filenames = [f['filename'] for f in file_data_list]
            sessao.add_turn(history_user_text, assistant_response, filenames=actual_filenames)
            sessao.save()

            result = {
                "response": assistant_response,
                "turn_count": sessao.turn_count,
                "context_usage": context_manager.get_context_usage_report(
                    sessao.get_context_size_chars()
                )
            }

            if processed_data:
                result["files_processed"] = processed_data.get("stats", {})

            return result

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama request timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no chat multimodal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/export")
async def export_session(session_id: str):
    """Exporta sessão em Markdown"""
    try:
        sessao = gerenciador.get_or_create(session_id=session_id)
        markdown_content = sessao.export_for_download()
        
        export_filename = f"medteacher_session_{session_id}.md"
        export_path = SESSIONS_DIR / session_id / export_filename
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return FileResponse(
            path=export_path,
            filename=export_filename,
            media_type="text/markdown"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro exportando sessão: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
