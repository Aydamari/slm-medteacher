"""
Gerenciamento de Prompts Mestres
Carrega e gerencia prompts em múltiplos idiomas
"""

from pathlib import Path
from typing import Dict, Optional
import logging

from backend.config import (
    PROMPT_TEMPLATES,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    PROMPTS_DIR
)

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Gerenciador de prompts mestres com suporte multilíngue
    Singleton pattern para cache em memória
    """
    
    _instance = None
    _cache: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _apply_prompt_repetition(self, prompt_text: str) -> str:
        """Dobra o system prompt (Leviathan et al., 2025).

        Técnica para modelos non-reasoning: repetir o system prompt compensa
        a atenção causal unidirecional, melhorando aderência às instruções.
        Usar APENAS em /chat (texto puro). Nunca em /chat/multimodal (contexto curto).
        """
        return f"{prompt_text}\n\n---\n\n{prompt_text}"

    def get_prompt(
        self,
        mode: str,
        language: str = DEFAULT_LANGUAGE,
        level: Optional[str] = None,
        apply_repetition: bool = False
    ) -> str:
        """
        Obtém prompt mestre para modo e idioma específicos.

        Args:
            mode: Modo de operação (clinical-reasoning, patient-communication)
            language: Código do idioma (en, pt)
            level: Não utilizado (compatibilidade)
            apply_repetition: Se True, dobra o prompt (Leviathan et al., 2025).
                              Usar apenas em /chat (texto). Nunca em /chat/multimodal.

        Returns:
            Texto do prompt mestre

        Raises:
            FileNotFoundError: Se prompt não encontrado
            ValueError: Se modo inválido
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Idioma '{language}' não suportado. Usando '{DEFAULT_LANGUAGE}'")
            language = DEFAULT_LANGUAGE

        if mode not in PROMPT_TEMPLATES:
            raise ValueError(f"Modo '{mode}' não reconhecido. Disponíveis: {list(PROMPT_TEMPLATES.keys())}")

        cache_key = f"{mode}_{language}_{level or 'default'}{'_rep' if apply_repetition else ''}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_path = self._get_prompt_path(mode, language, level)

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()

            if apply_repetition:
                prompt_text = self._apply_prompt_repetition(prompt_text)

            self._cache[cache_key] = prompt_text
            logger.info(f"Prompt carregado: {cache_key} ({len(prompt_text)} chars)")
            return prompt_text

        except FileNotFoundError:
            if language != DEFAULT_LANGUAGE:
                logger.warning(
                    f"Prompt não encontrado: {prompt_path}. "
                    f"Tentando fallback para '{DEFAULT_LANGUAGE}'"
                )
                return self.get_prompt(mode, DEFAULT_LANGUAGE, level)

            raise FileNotFoundError(
                f"Prompt mestre não encontrado: {prompt_path}\n"
                f"Verifique se o arquivo existe no diretório {PROMPTS_DIR}"
            )
    
    def _get_prompt_path(
        self,
        mode: str,
        language: str,
        level: Optional[str] = None
    ) -> Path:
        """
        Determina caminho do arquivo de prompt
        
        Args:
            mode: Modo de operação
            language: Código do idioma
            level: Nível do prompt (não usado - todos níveis usam mesmo prompt base)
        
        Returns:
            Path para o arquivo de prompt
        """
        filename = f"{mode}_{language}.txt"
        return PROMPTS_DIR / filename
    
    def reload_prompts(self):
        """Limpa cache e força recarregamento dos prompts"""
        self._cache.clear()
        logger.info("Cache de prompts limpo")
    
    def get_available_languages_for_mode(self, mode: str) -> list:
        """
        Retorna lista de idiomas disponíveis para um modo
        
        Args:
            mode: Modo de operação
        
        Returns:
            Lista de códigos de idioma disponíveis
        """
        if mode not in PROMPT_TEMPLATES:
            return []
        
        available = []
        for lang_code in SUPPORTED_LANGUAGES.keys():
            try:
                path = self._get_prompt_path(mode, lang_code)
                if path.exists():
                    available.append(lang_code)
            except Exception:
                continue
        
        return available
    
    def validate_all_prompts(self) -> Dict[str, Dict[str, bool]]:
        """
        Valida existência de todos os prompts esperados
        Útil para debugging e setup inicial
        
        Returns:
            Dicionário {modo: {idioma: existe}}
        """
        validation_report = {}
        
        for mode in PROMPT_TEMPLATES.keys():
            validation_report[mode] = {}
            
            for lang_code in SUPPORTED_LANGUAGES.keys():
                try:
                    path = self._get_prompt_path(mode, lang_code)
                    validation_report[mode][lang_code] = path.exists()
                except Exception as e:
                    validation_report[mode][lang_code] = False
                    logger.error(f"Erro validando {mode}_{lang_code}: {e}")
        
        return validation_report


prompt_manager = PromptManager()


def get_system_prompt(
    mode: str,
    language: str = DEFAULT_LANGUAGE,
    level: Optional[str] = None,
    apply_repetition: bool = False
) -> str:
    """Função helper para obter prompt mestre.

    apply_repetition=True: usar apenas em /chat (texto puro).
    apply_repetition=False: obrigatório em /chat/multimodal (contexto curto).
    """
    return prompt_manager.get_prompt(mode, language, level, apply_repetition)
