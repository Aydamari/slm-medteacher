import logging
from typing import List, Dict, Any
from paddleocr import PaddleOCR
import numpy as np
import cv2
import re

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Analisa PDFs e imagens de laudos usando PaddleOCR + Lógica de estruturação LLM-AIx.
    Focado em transformar OCR ruidoso em dados laboratoriais limpos.
    """
    def __init__(self, lang='pt'):
        # show_log foi removido nas versões > 3.0
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        logger.info(f"PaddleOCR inicializado para idioma: {lang}")

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Executa OCR e estruturação inspirada no LLM-AIx.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Falha ao decodificar imagem para OCR"}

            # 1. Extração de Texto (PaddleOCR)
            result = self.ocr.ocr(img, cls=True)
            
            extracted_lines = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.5: # Filtro de qualidade LLM-AIx
                        extracted_lines.append(text)

            # 2. Estruturação Heurística (Pré-processamento para o MedGemma)
            structured_data = self._structure_lab_results(extracted_lines)
            
            return {
                "status": "success",
                "full_text": "\n".join(extracted_lines),
                "structured_results": structured_data,
                "method": "PaddleOCR + LLM-AIx Structured Cleansing"
            }
            
        except Exception as e:
            logger.error(f"Erro no DocumentAnalyzer: {e}")
            return {"error": str(e)}

    def _structure_lab_results(self, lines: List[str]) -> List[Dict[str, str]]:
        """
        Lógica LLM-AIx para identificar pares Exame: Resultado.
        Busca padrões comuns em laudos (ex: Hemoglobina ..... 14.0 g/dL)
        """
        results = []
        # Regex para capturar exames comuns e valores numéricos
        pattern = re.compile(r'([a-zA-ZÀ-ÿ\s]+)[:\.]+\s*(\d+[\.,]?\d*)\s*([a-zA-Z/μµL]*)')
        
        for line in lines:
            match = pattern.search(line)
            if match:
                results.append({
                    "parameter": match.group(1).strip(),
                    "value": match.group(2).strip(),
                    "unit": match.group(3).strip()
                })
        return results

    def get_summary_for_prompt(self, analysis_results: Dict[str, Any]) -> str:
        if "error" in analysis_results:
            return f"[ERRO OCR: {analysis_results['error']}]"
        
        structured = analysis_results.get("structured_results", [])
        text = analysis_results.get("full_text", "")
        
        summary = "\n[CONTEÚDO DO LAUDO (Extraído e Estruturado)]:\n"
        if structured:
            summary += "VALORES DETECTADOS:\n"
            for item in structured:
                summary += f"- {item['parameter']}: {item['value']} {item['unit']}\n"
        else:
            summary += "TEXTO BRUTO (Nenhum padrão de valor detectado):\n"
            summary += text[:1000] # Limite para não estourar o prompt
            
        summary += "\n--- FIM DO DOCUMENTO ---\n"
        return summary

_document_analyzer = None
_document_analyzer_error = None

def get_document_analyzer() -> DocumentAnalyzer:
    """Lazy factory — instantiates DocumentAnalyzer on first call."""
    global _document_analyzer, _document_analyzer_error
    if _document_analyzer is None and _document_analyzer_error is None:
        try:
            _document_analyzer = DocumentAnalyzer()
        except Exception as e:
            _document_analyzer_error = str(e)
            logger.exception(f"Failed to initialize DocumentAnalyzer: {e}")
    if _document_analyzer_error is not None and _document_analyzer is None:
        raise RuntimeError(f"DocumentAnalyzer init failed: {_document_analyzer_error}")
    return _document_analyzer

def is_document_analyzer_loaded() -> bool:
    return _document_analyzer is not None

def document_analyzer_error():
    return _document_analyzer_error
