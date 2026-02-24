"""
MedGemma 1.5 Vision — Secondary Chest X-ray Interpreter

Uses MedGemma 1.5's native vision capability via Ollama to provide a
second interpretation of chest X-ray images, complementing the
TorchXRayVision ensemble. This allows the HAI-DEF model (MedGemma) to
contribute its own vision understanding directly, not only through text.

Returns None gracefully if Ollama is unavailable or the model does not
support vision — the TorchXRayVision analysis continues unaffected.
"""

import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Prompts tuned for MedGemma's medical training domain.
# Structured but concise — 512 token budget.
_PROMPT_EN = (
    "You are an AI trained on medical images. Carefully examine this chest X-ray and provide:\n"
    "1. Image quality and patient positioning (PA/AP, rotation, inspiration)\n"
    "2. Lung fields: describe any opacities, consolidations, effusions, or hyperinflation\n"
    "3. Cardiac silhouette: size (cardiothoracic ratio) and contour\n"
    "4. Mediastinum, hila, and bony structures (ribs, clavicles, spine)\n"
    "5. Overall impression in one sentence\n\n"
    "Be concise and clinically precise. Use standard radiological terminology."
)

_PROMPT_PT = (
    "Você é uma IA treinada em imagens médicas. Examine cuidadosamente esta radiografia de tórax e forneça:\n"
    "1. Qualidade da imagem e posicionamento do paciente (PA/AP, rotação, inspiração)\n"
    "2. Campos pulmonares: descreva opacidades, consolidações, derrames ou hiperinsuflação\n"
    "3. Silhueta cardíaca: tamanho (índice cardiotorácico) e contorno\n"
    "4. Mediastino, hilos e estruturas ósseas (costelas, clavículas, coluna)\n"
    "5. Impressão geral em uma frase\n\n"
    "Seja conciso e clinicamente preciso. Use terminologia radiológica padrão."
)


def analyze_with_medgemma_vision(
    image_bytes: bytes,
    model_id: str,
    ollama_base_url: str = "http://localhost:11434",
    language: str = "en",
    timeout: float = 90.0,
) -> Optional[str]:
    """
    Calls MedGemma 1.5 via Ollama's vision API to interpret a chest X-ray.

    Args:
        image_bytes: raw image bytes (JPEG, PNG, BMP, etc.)
        model_id:    Ollama model identifier
                     (e.g. "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M")
        ollama_base_url: Ollama API base URL (default: http://localhost:11434)
        language:    "en" or "pt" — selects prompt language
        timeout:     HTTP request timeout in seconds (default 90)

    Returns:
        Model's textual interpretation as a string, or None on any failure.
        Failure is always logged at WARNING level and never raises.
    """
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = _PROMPT_PT if language == "pt" else _PROMPT_EN

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{ollama_base_url}/api/chat", json=payload)

        if resp.status_code != 200:
            logger.warning(
                f"MedGemma vision: Ollama returned HTTP {resp.status_code} — "
                "native vision analysis skipped."
            )
            return None

        content = resp.json().get("message", {}).get("content", "").strip()
        if not content:
            logger.warning("MedGemma vision: empty response from model.")
            return None

        logger.info(
            f"MedGemma vision: received {len(content)} chars of X-ray interpretation "
            f"(model={model_id})."
        )
        return content

    except httpx.TimeoutException:
        logger.warning(
            f"MedGemma vision: request timed out after {timeout}s — skipping."
        )
        return None
    except Exception as e:
        logger.warning(
            f"MedGemma vision: {type(e).__name__}: {e} — skipping native vision."
        )
        return None
