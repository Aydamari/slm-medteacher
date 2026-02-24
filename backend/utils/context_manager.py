"""
Gerenciamento de Contexto Híbrido
Implementa estratégia de compressão inteligente de contexto
para manter conversações longas dentro da janela de tokens
"""

import re
from typing import List, Dict, Tuple
import logging

from backend.config import (
    MAX_CONTEXT_CHARS,
    CONTEXT_COMPRESSION_THRESHOLD,
    RECENT_TURNS_TO_KEEP_FULL,
    SUMMARY_TARGET_SIZE
)

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Gerenciador de contexto com compressão híbrida
    """
    
    def __init__(self):
        self.compression_threshold_chars = int(MAX_CONTEXT_CHARS * CONTEXT_COMPRESSION_THRESHOLD)
    
    def should_compress(self, current_context_size: int) -> bool:
        """
        Verifica se contexto precisa de compressão
        
        Args:
            current_context_size: Tamanho atual em caracteres
        
        Returns:
            True se deve comprimir
        """
        return current_context_size > self.compression_threshold_chars
    
    def compress_context(
        self,
        conversation_history: List[Dict],
        current_summary: str = ""
    ) -> Tuple[str, List[Dict]]:
        """
        Comprime contexto usando estratégia híbrida
        
        Estratégia:
        1. Mantém os N turnos mais recentes completos
        2. Resume turnos antigos em resumo consolidado
        3. Retorna (resumo_atualizado, turnos_recentes)
        
        Args:
            conversation_history: Lista completa de mensagens
            current_summary: Resumo consolidado existente
        
        Returns:
            (resumo_consolidado, mensagens_recentes)
        """
        if len(conversation_history) == 0:
            return current_summary, []
        
        messages_to_keep_full = RECENT_TURNS_TO_KEEP_FULL * 2
        
        if len(conversation_history) <= messages_to_keep_full:
            return current_summary, conversation_history
        
        old_messages = conversation_history[:-messages_to_keep_full]
        recent_messages = conversation_history[-messages_to_keep_full:]
        
        new_summary_content = self._summarize_messages(old_messages)
        
        if current_summary:
            consolidated_summary = self._merge_summaries(current_summary, new_summary_content)
        else:
            consolidated_summary = new_summary_content
        
        if len(consolidated_summary) > SUMMARY_TARGET_SIZE * 1.5:
            consolidated_summary = self._truncate_summary(consolidated_summary)
        
        logger.info(
            f"Contexto comprimido: {len(old_messages)} msgs → "
            f"resumo de {len(consolidated_summary)} chars + "
            f"{len(recent_messages)} msgs recentes"
        )
        
        return consolidated_summary, recent_messages
    
    def _summarize_messages(self, messages: List[Dict]) -> str:
        """
        Cria resumo estruturado de mensagens antigas com state tracking
        
        Args:
            messages: Lista de mensagens a resumir
        
        Returns:
            Texto resumido com estado clínico consolidado
        """
        clinical_state = self._extract_clinical_state(messages)
        
        summary_lines = [
            "=== ESTADO CLÍNICO CONSOLIDADO ===",
            ""
        ]

        if clinical_state["patient_info"]:
            summary_lines.append(f"Paciente: {clinical_state['patient_info']}")
        if clinical_state["chief_complaint"]:
            summary_lines.append(f"Queixa principal: {clinical_state['chief_complaint']}")
        if clinical_state["vital_signs"]:
            summary_lines.append(f"Sinais vitais (último): {clinical_state['vital_signs']}")
        if clinical_state["working_diagnoses"]:
            summary_lines.append(f"Hipóteses diagnósticas: {', '.join(clinical_state['working_diagnoses'])}")
        if clinical_state["tests_ordered"]:
            summary_lines.append(f"Exames solicitados: {', '.join(clinical_state['tests_ordered'])}")
        if clinical_state.get("tests_results"):
            summary_lines.append("Resultados de exames:")
            for r in clinical_state["tests_results"]:
                summary_lines.append(f"  - {r}")
        if clinical_state.get("medications_administered"):
            summary_lines.append(f"Medicações administradas: {'; '.join(clinical_state['medications_administered'])}")
        if clinical_state.get("procedures_performed"):
            summary_lines.append(f"Procedimentos realizados: {', '.join(clinical_state['procedures_performed'])}")
        if clinical_state.get("patient_evolution"):
            summary_lines.append("Evolução do paciente:")
            for e in clinical_state["patient_evolution"]:
                summary_lines.append(f"  - {e}")
        
        summary_lines.extend(["", "=== TURN-BY-TURN SUMMARY ===", ""])
        
        for i in range(0, len(messages), 2):
            if i + 1 >= len(messages):
                break
            
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            turn_summary = self._extract_turn_summary(user_msg["content"], assistant_msg["content"])
            
            if turn_summary:
                summary_lines.append(f"• {turn_summary}")
        
        summary_lines.append("")
        return "\n".join(summary_lines)
    
    def _extract_clinical_state(self, messages: List[Dict]) -> Dict[str, any]:
        """
        Extrai estado clínico consolidado do histórico

        Args:
            messages: Lista completa de mensagens

        Returns:
            Dicionário com estado clínico estruturado
        """
        state = {
            "patient_info": "",
            "chief_complaint": "",
            "vital_signs": "",
            "working_diagnoses": [],
            "tests_ordered": [],
            "tests_results": [],
            "medications_administered": [],
            "procedures_performed": [],
            "patient_evolution": []
        }

        combined_text = " ".join([msg.get("content", "") for msg in messages])

        patient_pattern = r'(?:Patient|Paciente):\s*([^,\n]+(?:,\s*\d+\s*(?:years|anos|años)[^,\n]*)?)'
        match = re.search(patient_pattern, combined_text, re.IGNORECASE)
        if match:
            state["patient_info"] = match.group(1).strip()

        complaint_pattern = r'(?:Chief Complaint|Queixa|Motivo):\s*([^\n]+)'
        match = re.search(complaint_pattern, combined_text, re.IGNORECASE)
        if match:
            state["chief_complaint"] = match.group(1).strip()

        vitals_keywords = ['PA:', 'FC:', 'FR:', 'BP:', 'HR:', 'RR:', 'Temp:', 'Tax:', 'SpO2:']
        for keyword in vitals_keywords:
            pattern = f'{re.escape(keyword)}\\s*([0-9/.°%]+(?:\\s*[a-zA-Z%°]*)?)'
            match = re.search(pattern, combined_text)
            if match:
                state["vital_signs"] += f"{keyword} {match.group(1).strip()} "

        dx_pattern = r'(?:Diagnos|Hipóte|Hypothes)[^\n:]*:\s*([^\n]+)'
        matches = re.findall(dx_pattern, combined_text, re.IGNORECASE)
        for match in matches[-3:]:
            diagnoses = [d.strip() for d in re.split(r'[,;]', match) if d.strip()]
            state["working_diagnoses"].extend(diagnoses)

        test_keywords = ['ECG', 'RX', 'Raio-X', 'X-ray', 'Hemograma', 'CBC', 'Troponina',
                        'Troponin', 'TC', 'CT', 'RM', 'MRI', 'Ultrassom', 'Ultrasound',
                        'Gasometria', 'Coagulograma', 'D-dímero']
        for keyword in test_keywords:
            if keyword.lower() in combined_text.lower():
                state["tests_ordered"].append(keyword)

        # Extract test results (lines with numeric values after test names)
        result_patterns = [
            r'(Hb[:\s]+[\d.,]+\s*g/dL[^.\n]*)',
            r'(Troponina[^.\n]{5,60})',
            r'(ECG:[^.\n]{10,150})',
            r'(Leucócitos[:\s]+[\d.,]+[^.\n]*)',
            r'(Creatinina[:\s]+[\d.,]+[^.\n]*)',
            r'(Glicemia[:\s]+[\d.,]+[^.\n]*)',
        ]
        for pat in result_patterns:
            match = re.search(pat, combined_text, re.IGNORECASE)
            if match:
                state["tests_results"].append(match.group(1).strip()[:120])

        # Extract medications administered
        med_pattern = r'(\w+\s+[\d.,]+\s*(?:mg|g|ml|mcg|UI)\s*(?:EV|IV|IM|VO|SC|SL)[^.\n]{0,80})'
        med_matches = re.findall(med_pattern, combined_text, re.IGNORECASE)
        state["medications_administered"] = list(dict.fromkeys(
            [m.strip()[:100] for m in med_matches]
        ))[:8]

        # Extract procedures performed
        proc_keywords = [
            'acesso venoso', 'intubação', 'sonda', 'SVD', 'SNG',
            'punção', 'drenagem', 'sutura', 'cardioversão', 'desfibrilação'
        ]
        for kw in proc_keywords:
            if kw.lower() in combined_text.lower():
                state["procedures_performed"].append(kw)

        # Extract patient evolution notes
        evol_pattern = r'[Pp]aciente\s+(refere|apresenta|evolu[ií]|mant[eé]m|nega)[^.]{5,100}\.'
        evol_matches = re.findall(evol_pattern, combined_text)
        if evol_matches:
            # Get the full match, not just the group
            for m in re.finditer(evol_pattern, combined_text):
                state["patient_evolution"].append(m.group(0).strip()[:120])
            state["patient_evolution"] = state["patient_evolution"][-3:]

        state["tests_ordered"] = list(set(state["tests_ordered"]))[:8]
        state["tests_results"] = state["tests_results"][:6]
        state["working_diagnoses"] = list(dict.fromkeys(state["working_diagnoses"]))[:3]

        return state
    
    def _extract_turn_summary(self, user_text: str, assistant_text: str) -> str:
        """
        Extrai informação clínica essencial de um turno
        
        Args:
            user_text: Mensagem do usuário
            assistant_text: Resposta do assistente
        
        Returns:
            String com informação clínica resumida
        """
        clinical_keywords = [
            'paciente', 'sintoma', 'diagnóstico', 'exame', 'hipótese', 
            'tratamento', 'conduta', 'medicação', 'histórico', 'quadro',
            'febre', 'dor', 'pressão', 'frequência', 'saturação',
            'patient', 'symptom', 'diagnosis', 'examination', 'hypothesis',
            'treatment', 'medication', 'history', 'condition',
            'fever', 'pain', 'pressure', 'rate', 'saturation',
            'paciente', 'síntoma', 'diagnóstico', 'examen', 'hipótesis',
            'tratamiento', 'medicación', 'historial', 'cuadro'
        ]
        
        combined = user_text + " " + assistant_text
        sentences = re.split(r'[.!?]\s+', combined)
        
        for sentence in sentences[:5]:
            if any(keyword in sentence.lower() for keyword in clinical_keywords):
                if len(sentence) > 200:
                    sentence = sentence[:197] + "..."
                return sentence
        
        user_words = user_text.split()[:15]
        return " ".join(user_words) + ("..." if len(user_words) == 15 else "")
    
    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """
        Combina resumo antigo com novo
        
        Args:
            old_summary: Resumo consolidado anterior
            new_summary: Novo resumo de mensagens recentes
        
        Returns:
            Resumo combinado
        """
        if len(old_summary) > SUMMARY_TARGET_SIZE:
            return new_summary
        
        return old_summary + "\n\n" + new_summary
    
    def _truncate_summary(self, summary: str) -> str:
        """
        Trunca resumo mantendo informações essenciais
        
        Args:
            summary: Resumo completo
        
        Returns:
            Resumo truncado
        """
        lines = summary.split('\n')
        
        if len(lines) <= 20:
            return summary
        
        header = lines[:5]
        recent = lines[-15:]
        
        truncated = header + ["", "... [histórico intermediário omitido] ...", ""] + recent
        return '\n'.join(truncated)
    
    def format_context_for_model(
        self,
        system_prompt: str,
        consolidated_summary: str,
        recent_messages: List[Dict],
        new_user_input: str
    ) -> List[Dict]:
        """
        Formata contexto completo para enviar ao modelo

        Args:
            system_prompt: Prompt mestre do sistema
            consolidated_summary: Resumo consolidado
            recent_messages: Mensagens recentes completas
            new_user_input: Nova entrada do usuário

        Returns:
            Lista de mensagens formatadas para Ollama
        """
        messages = []

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        if consolidated_summary:
            messages.append({
                "role": "system",
                "content": f"PREVIOUS CONVERSATION CONTEXT:\n\n{consolidated_summary}"
            })

        messages.extend(recent_messages)

        user_content = new_user_input

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages
    
    def estimate_context_tokens(self, messages: List[Dict]) -> int:
        """
        Estima número de tokens do contexto
        Aproximação: 1 token ≈ 4 caracteres
        
        Args:
            messages: Lista de mensagens
        
        Returns:
            Número estimado de tokens
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4
    
    def get_context_usage_report(
        self,
        current_context_size: int
    ) -> Dict[str, any]:
        """
        Gera relatório de uso do contexto
        
        Args:
            current_context_size: Tamanho atual em caracteres
        
        Returns:
            Dicionário com métricas
        """
        max_chars = MAX_CONTEXT_CHARS
        usage_percent = (current_context_size / max_chars) * 100
        
        return {
            "current_size_chars": current_context_size,
            "current_size_tokens": current_context_size // 4,
            "max_size_chars": max_chars,
            "max_size_tokens": max_chars // 4,
            "usage_percent": round(usage_percent, 1),
            "should_compress": self.should_compress(current_context_size),
            "status": self._get_status_label(usage_percent)
        }
    
    def _get_status_label(self, usage_percent: float) -> str:
        """Retorna label de status baseado no uso"""
        if usage_percent < 50:
            return "healthy"
        elif usage_percent < 80:
            return "moderate"
        elif usage_percent < 95:
            return "high"
        else:
            return "critical"


context_manager = ContextManager()
