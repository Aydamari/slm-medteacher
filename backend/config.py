"""
Configurações Centralizadas - SLM MedTeacher
Atualizado com suporte multilíngue
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from backend.utils.security import vault

# Carregar variáveis do arquivo .env (se existirem)
load_dotenv()

# ============================================================================
# CONFIGURAÇÕES DE API (CLOUD)
# ============================================================================
# OpenRouter: tenta primeiro o cofre seguro, depois o .env
OPENROUTER_API_KEY = vault.get_api_key() or os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# HuggingFace: elimina avisos de "unauthenticated requests" e habilita maior taxa de download
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN          # garante que subprocessos herdem
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN  # compatibilidade com versões antigas

# ============================================================================
# INFORMAÇÕES DO PRODUTO
# ============================================================================
PRODUCT_NAME = "SLM MedTeacher"
PRODUCT_VERSION = "1.0.0"
PRODUCT_DESCRIPTION = "Local AI Medical Teaching Assistant powered by MedGemma 1.5 4B"

# ============================================================================
# MULTILINGUISMO
# ============================================================================
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native_name": "English",
        "flag": "🇺🇸",
        "default": True  # Idioma padrão para competição
    },
    "pt": {
        "name": "Portuguese",
        "native_name": "Português",
        "flag": "🇧🇷"
    },
    # "es" removido: sem templates de prompt ES — fallback silencioso para EN confunde o usuário.
    # Adicionar clinical-reasoning_es.txt e patient-communication_es.txt para reabilitar.
}

DEFAULT_LANGUAGE = "en"

# Mensagens do sistema por idioma
SYSTEM_MESSAGES = {
    "en": {
        "welcome": "Welcome to MedTeacher! Select a mode to begin.",
        "session_created": "New session created. Ready to start.",
        "session_loaded": "Session loaded. Continuing from turn {turn_count}.",
        "max_turns": "Maximum turns ({max}) reached. Please start a new session.",
        "session_expired": "Session expired. Please start a new session.",
        "file_uploaded": "File uploaded successfully: {filename}",
        "processing": "Processing your request...",
        "error": "An error occurred. Please try again."
    },
    "pt": {
        "welcome": "Bem-vindo ao MedTeacher! Selecione um modo para começar.",
        "session_created": "Nova sessão criada. Pronto para começar.",
        "session_loaded": "Sessão carregada. Continuando do turno {turn_count}.",
        "max_turns": "Máximo de turnos ({max}) atingido. Inicie uma nova sessão.",
        "session_expired": "Sessão expirada. Inicie uma nova sessão.",
        "file_uploaded": "Arquivo enviado com sucesso: {filename}",
        "processing": "Processando sua solicitação...",
        "error": "Ocorreu um erro. Tente novamente."
    },
    # "es" removido: sem templates de prompt — fallback silencioso para EN confunde o usuário.
    # Adicionar clinical-reasoning_es.txt e patient-communication_es.txt para reabilitar.
}

# ============================================================================
# CAMINHOS DO PROJETO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
MODELFILES_DIR = PROJECT_ROOT / "modelfiles"

# Criar diretórios se não existirem
SESSIONS_DIR.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURAÇÕES DO OLLAMA
# ============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_TIMEOUT = 1200  # 20 minutos para hardware local e multimodal

# Mapeamento de Tiers para Modelos Reais
MODEL_TIERS = {
    "local_4b": {
        "name": "MedGemma 1.5 4B",
        "model_id": "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M",
        "type": "local"
    },
    "local_27b": {
        "name": "MedGemma 27B",
        "model_id": "thiagomoraes/medgemma-27b-it:Q4_K_M",
        "type": "local"
    },
    "llm_cloud": {
        "name": "LLMs (Cloud)",
        "type": "llm_cloud"
    },
    # Mantido para compatibilidade com sessões antigas
    "hybrid_cloud": {
        "name": "Hybrid Cloud (legado)",
        "type": "llm_cloud"
    }
}

# Modelos LLM disponíveis via OpenRouter
LLM_MODELS = {
    "gemini_2.5_flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "max_output_tokens": 8192,
        "temperature": 0.4,
        "vision": True,
    },
    "gemini_3_flash_preview": {
        "name": "Gemini 3 Flash Preview",
        "model_id": "google/gemini-3-flash-preview",
        "max_output_tokens": 8192,
        "temperature": 0.4,
        "vision": True,
    },
    "kimi_k2_thinking": {
        "name": "Kimi K2 Thinking",
        "model_id": "moonshotai/kimi-k2-thinking",
        "max_output_tokens": 8192,
        "temperature": 0.6,
        "vision": False,  # verificar suporte em openrouter.ai/models
    },
    "glm_5": {
        "name": "GLM-5 (Z.ai)",
        "model_id": "z-ai/glm-5",
        "max_output_tokens": 8192,
        "temperature": 0.4,
        "vision": False,  # verificar suporte em openrouter.ai/models
    },
}

# Mantido para compatibilidade, mas o uso real virá do TIER selecionado
MODELS_MAP = {
    "clinical-reasoning": "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M",
    "patient-communication": "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M"
}

# ============================================================================
# LIMITES DE CONTEXTO E UPLOADS
# ============================================================================
MAX_CONTEXT_TOKENS = 32000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4

# num_ctx enviado ao Ollama via options (deve caber na VRAM)
NUM_CTX_OLLAMA = 8192

# Uploads
MAX_FILE_SIZE_MB = 15
MAX_FILES_PER_REQUEST = 5
MAX_IMAGE_DIMENSION = 1024  # Aumentado para 1024 para máxima precisão na digitalização

# Turnos
MAX_TURNS_PER_SESSION = 20

# ============================================================================
# CONTEXTO HÍBRIDO
# ============================================================================
CONTEXT_COMPRESSION_THRESHOLD = 0.8
RECENT_TURNS_TO_KEEP_FULL = 5
SUMMARY_TARGET_SIZE = 2000

# ============================================================================
# PROCESSAMENTO DE IMAGENS/DOCUMENTOS
# ============================================================================
ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp"]
ALLOWED_DOC_FORMATS = ["pdf"]
IMAGE_COMPRESSION_QUALITY = 85

# ============================================================================
# SISTEMA DE SESSÕES
# ============================================================================
SESSION_ID_LENGTH = 16
SESSION_EXPIRY_HOURS = 24

# ============================================================================
# PROMPTS MESTRES (MULTILÍNGUES)
# ============================================================================
# Estrutura: {modo}_{idioma}.txt
# Ex: clinical-reasoning_en.txt, clinical-reasoning_pt.txt
PROMPT_TEMPLATES = {
    "clinical-reasoning": {
        "en": PROMPTS_DIR / "clinical-reasoning_en.txt",
        "pt": PROMPTS_DIR / "clinical-reasoning_pt.txt",
    },
    "patient-communication": {
        "en": PROMPTS_DIR / "patient-communication_en.txt",
        "pt": PROMPTS_DIR / "patient-communication_pt.txt",
    }
}

# ============================================================================
# MENSAGENS DE ERRO (MULTILÍNGUES)
# ============================================================================
ERROR_MESSAGES = {
    "en": {
        "file_too_large": f"File too large. Maximum: {MAX_FILE_SIZE_MB}MB",
        "too_many_files": f"Too many files. Maximum: {MAX_FILES_PER_REQUEST} at once",
        "invalid_format": "Unsupported file format",
        "context_overflow": "Context too long. Start new session.",
        "session_expired": "Session expired. Start new session.",
        "session_not_found": "Session not found.",
        "ollama_error": "Error communicating with Ollama. Check if it's running.",
        "model_not_found": "Model not found. Run installer.",
        "max_turns_reached": f"Turn limit ({MAX_TURNS_PER_SESSION}) reached. Start new session."
    },
    "pt": {
        "file_too_large": f"Arquivo muito grande. Máximo: {MAX_FILE_SIZE_MB}MB",
        "too_many_files": f"Muitos arquivos. Máximo: {MAX_FILES_PER_REQUEST} por vez",
        "invalid_format": "Formato de arquivo não suportado",
        "context_overflow": "Contexto muito longo. Inicie nova sessão.",
        "session_expired": "Sessão expirada. Inicie nova sessão.",
        "session_not_found": "Sessão não encontrada.",
        "ollama_error": "Erro ao comunicar com Ollama. Verifique se está rodando.",
        "model_not_found": "Modelo não encontrado. Execute o instalador.",
        "max_turns_reached": f"Limite de {MAX_TURNS_PER_SESSION} turnos atingido. Inicie nova sessão."
    },
}

# ============================================================================
# DESENVOLVIMENTO
# ============================================================================
DEBUG = False
CORS_ORIGINS = ["*"]  # Permitir todas as origens (desenvolvimento)
