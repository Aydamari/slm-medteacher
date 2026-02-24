"""
Detector de Intenção para Modo Raciocínio Clínico
Simplificado: Apenas limpa e prepara a mensagem sem injeção de modos acadêmicos.
"""

def detect_clinical_intent(message: str, has_files: bool = False) -> str:
    """Retorna um valor padrão já que não usamos mais modos ramificados."""
    return "CLINICAL"

def inject_intent_tag(message: str, intent: str) -> str:
    """Apenas formata a mensagem para o modelo sem tags de modo."""
    return message.strip()
