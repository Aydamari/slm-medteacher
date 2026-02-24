"""
ECG Digitizer Utility - Versão Simplificada e Robusta
Foca na calibração e análise de ritmo, sem segmentação anatômica incerta.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ECGDigitizer:
    def __init__(self):
        self.px_per_mv = 50.0 
        self.px_per_ms = 0.25 
        self.calibrated = False

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Falha ao decodificar imagem"}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 1. Tentar Calibração pelo pulso padrão
            self._calibrate_from_pulse(binary)

            # 2. Análise de Ritmo Global (DII Longo na base)
            rhythm_info = self._analyze_rhythm(binary)

            return {
                "status": "success",
                "extracted_data": {
                    "calibration": "Detectada" if self.calibrated else "Padrão (Estimada)",
                    "scale": f"{self.px_per_mv} px/mV",
                    "rhythm": rhythm_info,
                    "resolution": f"{img.shape[1]}x{img.shape[0]}"
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na digitalização simplificada: {e}")
            return {"error": str(e)}

    def _calibrate_from_pulse(self, binary_img: np.ndarray):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                if 1.5 < (h/w) < 2.5 and h > 20: 
                    self.px_per_mv = h / 1.0
                    self.px_per_ms = w / 200.0
                    self.calibrated = True
                    return

    def _analyze_rhythm(self, binary_img: np.ndarray) -> Dict[str, Any]:
        h, w = binary_img.shape
        roi_rhythm = binary_img[int(h*0.80):h, :] # Últimos 20% da imagem
        
        projection = np.sum(roi_rhythm, axis=0)
        peaks = []
        in_peak = False
        for i, val in enumerate(projection):
            if val > (np.max(projection) * 0.5) and not in_peak:
                peaks.append(i)
                in_peak = True
            elif val < (np.max(projection) * 0.4):
                in_peak = False
        
        if len(peaks) > 3:
            rr = np.diff(peaks)
            cv = (np.std(rr) / np.mean(rr)) * 100
            hr = int(60 / (np.mean(rr) * (1.0 / (self.px_per_ms * 1000)))) if self.px_per_ms > 0 else 0
            return {
                "regularity": "IRREGULAR (Atenção: Possível Arritmia/FA)" if cv > 15 else "REGULAR",
                "hr": hr,
                "cv": round(cv, 1)
            }
        return {"regularity": "Análise visual necessária"}

    def get_summary_for_prompt(self, analysis_results: Dict[str, Any]) -> str:
        if "error" in analysis_results: return f"[AVISO: Falha no preprocessamento: {analysis_results['error']}]"
        
        data = analysis_results.get("extracted_data", {})
        rhythm = data.get("rhythm", {})
        
        return (
            "\n[TELEMETRIA PRÉ-ANALISADA]:\n"
            f"- Calibração: {data.get('calibration')} ({data.get('scale')})\n"
            f"- Ritmo na derivação longa: {rhythm.get('regularity')}\n"
            f"- Frequência Cardíaca estimada: {rhythm.get('hr')} bpm\n"
            "- NOTA: Analise as 12 derivações na imagem para identificar desníveis de ST ou outras patologias.\n"
        )

ecg_digitizer = ECGDigitizer()
