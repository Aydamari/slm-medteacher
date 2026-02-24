"""
Gerenciamento de Sessões
Responsável por criar, carregar, salvar e gerenciar sessões de usuário
"""

import json
import shutil
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backend.config import (
    SESSIONS_DIR,
    SESSION_ID_LENGTH,
    SESSION_EXPIRY_HOURS,
    MAX_TURNS_PER_SESSION,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)


class Sessao:
    """
    Representa uma sessão de usuário com histórico de conversação
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        mode: str = "clinical-reasoning",
        level: str = "ciclo-clinico",
        model: str = "thiagomoraes/medgemma-1.5-4b-it:Q8_0",
        language: str = "en",
        model_tier: str = "local_4b",
        llm_model: str = "gemini_2.5_flash"
    ):
        """
        Args:
            session_id: ID único da sessão (gera novo se None)
            mode: Modo de operação
            level: Nível do prompt
            model: Modelo Ollama (usado apenas em tiers locais)
            language: Idioma da sessão
            model_tier: Categoria do modelo (local_4b, local_27b, llm_cloud)
            llm_model: Modelo LLM específico quando model_tier == llm_cloud
        """
        self.session_id = session_id or self._generate_session_id()
        self.mode = mode
        self.level = level
        self.model = model
        self.language = language
        self.model_tier = model_tier
        self.llm_model = llm_model
        self.created_at = datetime.now().isoformat()
        self.last_accessed = datetime.now().isoformat()
        
        self.conversation_history: List[Dict] = []
        self.consolidated_summary = ""
        self.turn_count = 0
        self.total_tokens_approx = 0
        self.metadata = {}
        
        self.session_dir = SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def _generate_session_id() -> str:
        """Gera ID único de sessão"""
        return uuid.uuid4().hex[:SESSION_ID_LENGTH]
    
    def add_turn(self, user_input: str, assistant_output: str, filenames: Optional[List[str]] = None):
        """
        Adiciona um turno completo à conversação
        
        Args:
            user_input: Mensagem do usuário
            assistant_output: Resposta do assistente
            filenames: Lista de nomes de arquivos anexados (opcional)
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "filenames": filenames or [],
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_output,
            "timestamp": datetime.now().isoformat()
        })
        
        self.turn_count += 1
        self.last_accessed = datetime.now().isoformat()
        
        turn_chars = len(user_input) + len(assistant_output)
        self.total_tokens_approx += turn_chars // 4
    
    def get_full_history(self) -> List[Dict]:
        """Retorna histórico completo da conversação"""
        return self.conversation_history
    
    def get_recent_turns(self, n: int = 2) -> List[Dict]:
        """
        Retorna os N turnos mais recentes completos
        
        Args:
            n: Número de turnos (par user+assistant) a retornar
        """
        messages_to_get = n * 2
        return self.conversation_history[-messages_to_get:] if len(self.conversation_history) >= messages_to_get else self.conversation_history
    
    def is_expired(self) -> bool:
        """Verifica se sessão expirou"""
        last_access = datetime.fromisoformat(self.last_accessed)
        expiry_time = last_access + timedelta(hours=SESSION_EXPIRY_HOURS)
        return datetime.now() > expiry_time
    
    def can_continue(self) -> bool:
        """Verifica se sessão pode aceitar mais turnos"""
        return self.turn_count < MAX_TURNS_PER_SESSION
    
    def save(self):
        """Salva sessão em disco"""
        metadata_file = self.session_dir / "metadata.json"
        history_file = self.session_dir / "history.jsonl"
        summary_file = self.session_dir / "summary.txt"
        
        metadata = {
            "session_id": self.session_id,
            "mode": self.mode,
            "level": self.level,
            "model": self.model,
            "language": self.language,
            "model_tier": self.model_tier,
            "llm_model": self.llm_model,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "turn_count": self.turn_count,
            "total_tokens_approx": self.total_tokens_approx,
            "metadata": self.metadata
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            for message in self.conversation_history:
                f.write(json.dumps(message, ensure_ascii=False) + '\n')
        
        if self.consolidated_summary:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(self.consolidated_summary)
    
    @classmethod
    def load(cls, session_id: str) -> Optional['Sessao']:
        """
        Carrega sessão do disco
        
        Args:
            session_id: ID da sessão a carregar
        
        Returns:
            Objeto Sessao ou None se não encontrada
        """
        session_dir = SESSIONS_DIR / session_id
        metadata_file = session_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        sessao = cls(
            session_id=metadata["session_id"],
            mode=metadata["mode"],
            level=metadata["level"],
            model=metadata["model"],
            language=metadata.get("language", "en"),
            model_tier=metadata.get("model_tier", "local_4b"),
            llm_model=metadata.get("llm_model", "gemini_2.5_flash")
        )
        
        sessao.created_at = metadata["created_at"]
        sessao.last_accessed = metadata["last_accessed"]
        sessao.turn_count = metadata["turn_count"]
        sessao.total_tokens_approx = metadata["total_tokens_approx"]
        sessao.metadata = metadata.get("metadata", {})
        
        history_file = session_dir / "history.jsonl"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                sessao.conversation_history = [
                    json.loads(line) for line in f if line.strip()
                ]
        
        summary_file = session_dir / "summary.txt"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                sessao.consolidated_summary = f.read()
        
        return sessao
    
    def export_for_download(self) -> str:
        """
        Exporta sessão completa em formato Markdown para download
        
        Returns:
            String em Markdown com toda a conversação
        """
        lines = [
            f"# Sessão de Estudo - {self.mode.title()}",
            f"**Data**: {datetime.fromisoformat(self.created_at).strftime('%d/%m/%Y %H:%M')}",
            f"**Turnos**: {self.turn_count}",
            "",
            "---",
            ""
        ]
        
        for i, msg in enumerate(self.conversation_history):
            role = "**Você**" if msg["role"] == "user" else "**MedTeacher**"
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime('%H:%M')
            
            lines.append(f"### {role} ({timestamp})")
            
            # Adicionar arquivos se existirem
            filenames = msg.get("filenames", [])
            if filenames:
                masked_files = [self._mask_filename(f) for f in filenames]
                lines.append(f"> 📎 Arquivos: {', '.join(masked_files)}")
                lines.append("")

            lines.append(msg["content"])
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)

    @staticmethod
    def _mask_filename(filename: str) -> str:
        """Anonimiza nome do arquivo para o export"""
        if not filename: return ""
        parts = filename.split('.')
        ext = parts.pop()
        name = ".".join(parts)
        if len(name) <= 6: return f"***.{ext}"
        return f"{name[:3]}...{name[-3:]}.{ext}"
    
    def get_context_size_chars(self) -> int:
        """Retorna tamanho atual do contexto em caracteres"""
        total_chars = len(self.consolidated_summary)
        for msg in self.conversation_history:
            total_chars += len(msg["content"])
        return total_chars
    
    def __repr__(self):
        return f"Sessao(id={self.session_id}, mode={self.mode}, turns={self.turn_count})"


class GerenciadorSessoes:
    """
    Gerenciador central de todas as sessões
    Singleton pattern para cache em memória
    """
    
    _instance = None
    _cache: Dict[str, Sessao] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_or_create(
        self,
        session_id: Optional[str] = None,
        mode: str = "clinical-reasoning",
        level: str = "ciclo-clinico",
        model: str = "thiagomoraes/medgemma-1.5-4b-it:Q8_0",
        language: str = "en",
        model_tier: str = "local_4b",
        llm_model: str = "gemini_2.5_flash"
    ) -> Sessao:
        """
        Obtém sessão existente ou cria nova
        """
        if session_id is None:
            sessao = Sessao(
                mode=mode, level=level, model=model,
                language=language, model_tier=model_tier, llm_model=llm_model
            )
            self._cache[sessao.session_id] = sessao
            return sessao
        
        if session_id in self._cache:
            sessao = self._cache[session_id]
            if sessao.is_expired():
                del self._cache[session_id]
                raise ValueError(ERROR_MESSAGES["en"]["session_expired"])
            sessao.last_accessed = datetime.now().isoformat()
            return sessao
        
        sessao = Sessao.load(session_id)
        if sessao is None:
            raise ValueError(ERROR_MESSAGES["en"]["session_not_found"])
        
        if sessao.is_expired():
            raise ValueError(ERROR_MESSAGES["en"]["session_expired"])
        
        self._cache[session_id] = sessao
        sessao.last_accessed = datetime.now().isoformat()
        return sessao
    
    def save_session(self, session_id: str):
        """Salva sessão em disco"""
        if session_id in self._cache:
            self._cache[session_id].save()
    
    def cleanup_expired(self):
        """Remove sessões expiradas do cache em memória"""
        expired_ids = [
            sid for sid, sessao in self._cache.items()
            if sessao.is_expired()
        ]
        for sid in expired_ids:
            del self._cache[sid]
        return expired_ids

    def cleanup_expired_from_disk(self) -> int:
        """
        Remove sessões expiradas do disco.
        Preserva arquivos .md (exports) movendo-os para sessions/exports/.
        Remove todo o resto do diretório da sessão.

        Returns:
            Número de sessões limpas
        """
        if not SESSIONS_DIR.exists():
            return 0

        exports_dir = SESSIONS_DIR / "exports"
        cleaned = 0

        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            if session_dir.name == "exports":
                continue

            metadata_file = session_dir / "metadata.json"
            if not metadata_file.exists():
                # Diretório órfão sem metadata — limpar
                self._remove_session_dir(session_dir, exports_dir)
                cleaned += 1
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                last_accessed = datetime.fromisoformat(metadata.get("last_accessed", "2000-01-01"))
                expiry_time = last_accessed + timedelta(hours=SESSION_EXPIRY_HOURS)

                if datetime.now() > expiry_time:
                    self._remove_session_dir(session_dir, exports_dir)
                    # Also remove from cache if present
                    self._cache.pop(session_dir.name, None)
                    cleaned += 1

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Metadata corrompido em {session_dir.name}, limpando: {e}")
                self._remove_session_dir(session_dir, exports_dir)
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleanup: {cleaned} sessões expiradas removidas do disco")

        return cleaned

    @staticmethod
    def _remove_session_dir(session_dir: Path, exports_dir: Path):
        """
        Remove diretório de sessão preservando arquivos .md (exports).
        """
        # Salvar exports antes de apagar
        md_files = list(session_dir.glob("*.md"))
        if md_files:
            exports_dir.mkdir(exist_ok=True)
            for md_file in md_files:
                dest = exports_dir / md_file.name
                try:
                    shutil.move(str(md_file), str(dest))
                    logger.info(f"Export preservado: {md_file.name} -> exports/")
                except Exception as e:
                    logger.warning(f"Erro movendo export {md_file.name}: {e}")

        # Remover diretório inteiro
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            logger.error(f"Erro removendo diretório {session_dir.name}: {e}")
