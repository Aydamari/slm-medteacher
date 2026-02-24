import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# #3: Frontend → Backend exam type normalization
EXAM_TYPE_MAP = {
    "rx": "xray",
    "tc": "ct",
    "lab": "lab_results",
    # These are already correct but included for completeness
    "ecg": "ecg",
    "general": "general",
}

# Extension-based exam type hints for per-file routing (#4)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_PDF_EXTENSIONS = {".pdf"}


def _detect_exam_type_for_file(filename: str, content_type: str, hint: str) -> str:
    """
    #4: Detect real exam_type per file via MIME/extension.
    Uses the session hint only when auto-detection is inconclusive.
    """
    ext = Path(filename).suffix.lower()

    # PDFs are almost always lab results / documents
    if ext in _PDF_EXTENSIONS or content_type == "application/pdf":
        return "lab_results"

    # If we have a concrete hint from the frontend, use it (already normalised)
    if hint and hint not in ("general",):
        return hint

    return "general"


class PerceptionManager:
    """
    Gerenciador central de ferramentas de percepção (CNNs e OCR).
    Decide qual ferramenta usar baseada no tipo de arquivo e contexto.
    Analyzers are loaded lazily on first use (#10).
    """

    def _get_document_analyzer(self):
        from .document_analyzer import get_document_analyzer
        return get_document_analyzer()

    def _get_ecg_analyzer(self):
        from .ecg_analyzer import get_ecg_analyzer
        return get_ecg_analyzer()

    def _get_xray_analyzer(self):
        from .xray_analyzer import get_xray_analyzer
        return get_xray_analyzer()

    @staticmethod
    def normalize_exam_type(raw: str) -> str:
        """#3: Normalise frontend exam type strings to backend values."""
        return EXAM_TYPE_MAP.get(raw, raw)

    def analyze_file(
        self,
        content: bytes,
        filename: str,
        exam_type: str = "general",
        content_type: str = "",
        language: str = "en",
    ) -> str:
        """
        Analisa um arquivo e retorna um resumo textual para o prompt.

        Args:
            content: raw bytes of the uploaded file
            filename: original file name
            exam_type: already-normalised exam type
            content_type: MIME type from the upload
            language: user language for localised fallback messages (#8)
        """
        summary = ""

        # Resolve actual exam type per file (#4)
        resolved_type = _detect_exam_type_for_file(filename, content_type, exam_type)

        # 1. Document / Lab results
        is_pdf = filename.lower().endswith('.pdf') or content_type == "application/pdf"

        if is_pdf:
            # PDFs → PyPDF2 text extraction (cv2.imdecode cannot read PDF bytes)
            try:
                from backend.utils.pdf_extractor import pdf_extractor
                doc = pdf_extractor.extract_text(content, filename)
                extracted = doc.get("text", "").strip()
                page_count = len(doc.get("pages", []))
                if extracted:
                    summary = (
                        f"\n[CONTEÚDO DO LAUDO / DOCUMENTO (PDF — {page_count} pág.)]:\n"
                        + extracted[:4000]
                        + ("\n[... conteúdo truncado ...]" if len(extracted) > 4000 else "")
                        + "\n---\n"
                    )
                else:
                    summary = "\n[PDF recebido mas sem texto extraível — pode ser PDF de imagem escaneada]\n"
            except Exception as e:
                logger.error(f"Falha ao extrair texto do PDF '{filename}': {e}")
                summary = f"\n[PDF '{filename}' não pôde ser lido — arquivo corrompido ou formato inválido]\n"

        elif resolved_type == "lab_results":
            # Lab result image → PaddleOCR
            analyzer = self._get_document_analyzer()
            analysis = analyzer.analyze(content)
            summary = analyzer.get_summary_for_prompt(analysis)

        # 2. ECG
        elif resolved_type == "ecg":
            analyzer = self._get_ecg_analyzer()
            analysis = analyzer.analyze(content)
            summary = analyzer.get_summary_for_prompt(analysis, language=language)

        # 3. Radiografia de Tórax (RX)
        # TorchXRayVision foi treinado exclusivamente em RX de tórax — não aplicar a TC ou RM.
        elif resolved_type == "xray":
            analyzer = self._get_xray_analyzer()
            analysis = analyzer.analyze(content)
            summary = analyzer.get_summary_for_prompt(analysis, language=language)

            # MedGemma 1.5 native vision — HAI-DEF model direct interpretation.
            # Supplements TorchXRayVision ensemble with MedGemma's medical vision training.
            # Skipped silently if Ollama is unavailable or model not loaded.
            try:
                from backend.config import OLLAMA_BASE_URL, MODEL_TIERS
                from .medgemma_vision import analyze_with_medgemma_vision
                mg_model_id = MODEL_TIERS.get("local_4b", {}).get("model_id", "")
                if mg_model_id:
                    mg_text = analyze_with_medgemma_vision(
                        content, mg_model_id, OLLAMA_BASE_URL, language=language
                    )
                    if mg_text:
                        if language == "pt":
                            summary += (
                                f"\n[INTERPRETAÇÃO NATIVA MEDGEMMA 1.5 (Visão IA)]:\n"
                                f"{mg_text}\n---\n"
                            )
                        else:
                            summary += (
                                f"\n[MEDGEMMA 1.5 NATIVE VISION INTERPRETATION]:\n"
                                f"{mg_text}\n---\n"
                            )
            except Exception as _mg_exc:
                logger.debug(f"MedGemma vision skipped: {_mg_exc}")

        # 4. TC e RM — sem modelo ML compatível disponível.
        # O LLM fará análise visual direta; exam_instructions["tc"] fornece a instrução.
        elif resolved_type in ("ct", "mri"):
            if language == "pt":
                summary = (
                    "\n[ANÁLISE AUTOMÁTICA (IA) NÃO DISPONÍVEL PARA TC/RM]\n"
                    "Prossiga com análise visual direta da imagem.\n"
                    "---\n"
                )
            else:
                summary = (
                    "\n[AUTOMATED AI ANALYSIS NOT AVAILABLE FOR CT/MRI]\n"
                    "Proceed with direct visual analysis of the image.\n"
                    "---\n"
                )

        return summary

    # --- Readiness introspection for /health (#11) ---

    @staticmethod
    def get_readiness() -> Dict[str, Any]:
        # E11: Cada componente em try/except próprio — falha de import de um não silencia os outros
        result: Dict[str, Any] = {}

        try:
            from .ecg_analyzer import is_ecg_analyzer_loaded, ecg_analyzer_error
            result["ecg"] = {"loaded": is_ecg_analyzer_loaded(), "error": ecg_analyzer_error()}
        except Exception as exc:
            result["ecg"] = {"loaded": False, "error": f"import error: {exc}"}

        try:
            from .xray_analyzer import is_xray_analyzer_loaded, xray_analyzer_error
            result["xray"] = {"loaded": is_xray_analyzer_loaded(), "error": xray_analyzer_error()}
        except Exception as exc:
            result["xray"] = {"loaded": False, "error": f"import error: {exc}"}

        try:
            from .document_analyzer import is_document_analyzer_loaded, document_analyzer_error
            result["ocr"] = {"loaded": is_document_analyzer_loaded(), "error": document_analyzer_error()}
        except Exception as exc:
            result["ocr"] = {"loaded": False, "error": f"import error: {exc}"}

        return result


perception_manager = PerceptionManager()
