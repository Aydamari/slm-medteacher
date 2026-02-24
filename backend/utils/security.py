import logging
import os
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# Vault file stored outside the project directory for security
VAULT_FILE = Path.home() / ".medteacher" / "vault.enc"
_VAULT_KEY_FILE = Path.home() / ".medteacher" / "vault.key"

class MedicalAnonymizer:
    """
    Sistema de anonimização baseado em Microsoft Presidio.
    Configurado explicitamente para Spacy PT/EN.
    """
    def __init__(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_analyzer.nlp_engine import SpacyNlpEngine

            # CORREÇÃO: Usar 'lang_code' em vez de 'lang' para conformidade com Presidio
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "pt", "model_name": "pt_core_news_lg"},
                    {"lang_code": "en", "model_name": "en_core_web_lg"}
                ]
            }
            nlp_engine = SpacyNlpEngine(models=configuration["models"])

            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
            self.anonymizer = AnonymizerEngine()
            logger.info("Sistema de Anonimização Presidio inicializado corretamente.")
        except Exception as e:
            logger.error(f"Erro inicializando Presidio: {e}")
            self.analyzer = None
            self.anonymizer = None

    def anonymize(self, text: str, language: str = "pt") -> str:
        if not text or not self.analyzer:
            return text

        try:
            target_lang = "pt" if language == "pt" else "en"
            # EMAIL_ADDRESS não tem recognizer nativo para PT — omitir para evitar warning
            entities = (
                ["PERSON", "DATE_TIME", "LOCATION", "PHONE_NUMBER"]
                if target_lang == "pt"
                else ["PERSON", "DATE_TIME", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS"]
            )
            results = self.analyzer.analyze(
                text=text,
                language=target_lang,
                entities=entities,
            )

            from presidio_anonymizer.entities import OperatorConfig
            operators = {
                "PERSON": OperatorConfig("replace", {"new_value": "[PACIENTE]"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATA]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[LOCAL]"}),
            }

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )

            return anonymized_result.text
        except Exception as e:
            logger.error(f"Erro durante anonimização: {e}")
            return text

medical_anonymizer = MedicalAnonymizer()

class Vault:
    """
    Cofre de chaves com dois mecanismos (em ordem de prioridade):

    1. Fernet (primário): AES-128-CBC + HMAC-SHA256.
       Arquivos: vault.enc (texto cifrado) + vault.key (chave simétrica, chmod 600).
       Usado para novas chaves salvas via setup_secrets.py.

    2. Deceptive Vault (fallback): chave distribuída em 30 000 chars de ruído ASCII
       usando posições primas determinísticas (salt=42, step=7).
       Formato: vault.bin — compatibilidade com instalações anteriores.
       NÃO gera novos vault.bin; só lê se vault.enc estiver ausente.
    """

    # Parâmetros do Deceptive Vault (deve ser idêntico ao setup original)
    _DV_SALT = 42
    _DV_STEP = 7
    _DV_MASK = 30_000
    _DV_FILE = VAULT_FILE.parent / "vault.bin"

    def __init__(self):
        self._fernet: Optional[Fernet] = None

    def _ensure_dir(self):
        VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _get_fernet(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        self._ensure_dir()
        if _VAULT_KEY_FILE.exists():
            key = _VAULT_KEY_FILE.read_bytes()
        else:
            key = Fernet.generate_key()
            _VAULT_KEY_FILE.write_bytes(key)
            os.chmod(_VAULT_KEY_FILE, 0o600)
        self._fernet = Fernet(key)
        return self._fernet

    def encrypt_and_save(self, api_key: str) -> bool:
        """Cifra e salva a chave no formato Fernet (vault.enc + vault.key)."""
        f = self._get_fernet()
        encrypted = f.encrypt(api_key.encode("utf-8"))
        self._ensure_dir()
        VAULT_FILE.write_bytes(encrypted)
        os.chmod(VAULT_FILE, 0o600)
        logger.info("API key encrypted and saved to vault (Fernet).")
        return True

    def _read_deceptive_vault(self) -> Optional[str]:
        """Lê vault.bin usando o algoritmo Deceptive Vault (fallback)."""
        if not self._DV_FILE.exists():
            return None
        try:
            noise = self._DV_FILE.read_text(encoding="utf-8")
            if len(noise) < self._DV_MASK:
                return None
            key_len = ord(noise[self._DV_SALT]) - 33
            if key_len <= 0 or key_len > 512:
                return None
            chars = []
            for i in range(key_len):
                pos = self._DV_SALT + 10 + (i * self._DV_STEP)
                chars.append(noise[pos])
            api_key = "".join(chars)
            if not api_key.strip():
                return None
            return api_key
        except Exception as e:
            logger.warning(f"Deceptive vault read failed: {e}")
            return None

    def get_api_key(self) -> Optional[str]:
        """
        Tenta obter a chave Gemini em duas etapas:
        1. vault.enc (Fernet) — primário
        2. vault.bin (Deceptive) — fallback para instalações antigas
        """
        # 1. Fernet
        if VAULT_FILE.exists():
            try:
                f = self._get_fernet()
                encrypted = VAULT_FILE.read_bytes()
                key = f.decrypt(encrypted).decode("utf-8")
                if key:
                    return key
            except Exception as e:
                logger.warning(f"Fernet vault read failed: {e}")

        # 2. Deceptive Vault (vault.bin — retrocompatibilidade)
        key = self._read_deceptive_vault()
        if key:
            logger.info("API key loaded from legacy deceptive vault (vault.bin).")
            return key

        return None

vault = Vault()
