"""
Extrator de Texto de PDFs
Processa documentos PDF e extrai texto estruturado
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, List
from PyPDF2 import PdfReader

from backend.config import MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extrator de texto de documentos PDF
    Otimizado para laudos médicos e documentos clínicos
    """
    
    def __init__(self):
        self.max_file_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    
    def extract_text(
        self,
        file_content: bytes,
        filename: str = "document.pdf"
    ) -> Dict[str, any]:
        """
        Extrai texto de arquivo PDF
        
        Args:
            file_content: Bytes do arquivo PDF
            filename: Nome do arquivo (para logging)
        
        Returns:
            Dicionário com:
                - text: Texto completo extraído
                - pages: Lista de textos por página
                - metadata: Metadados do PDF
                - stats: Estatísticas (páginas, chars, etc)
        
        Raises:
            ValueError: Se arquivo inválido ou muito grande
        """
        # Validar tamanho
        if len(file_content) > self.max_file_size_bytes:
            raise ValueError(
                f"PDF too large: {len(file_content) / 1024 / 1024:.1f}MB. "
                f"Maximum: {MAX_FILE_SIZE_MB}MB"
            )
        
        try:
            # Criar objeto PdfReader a partir de bytes
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            # Extrair metadados
            metadata = self._extract_metadata(reader)
            
            # Extrair texto página por página
            pages_text = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append({
                            "page_number": page_num,
                            "text": page_text.strip()
                        })
                except Exception as e:
                    logger.warning(f"Erro extraindo página {page_num} de {filename}: {e}")
                    pages_text.append({
                        "page_number": page_num,
                        "text": f"[Erro ao extrair página {page_num}]"
                    })
            
            # Combinar todo o texto
            full_text = "\n\n".join([p["text"] for p in pages_text])
            
            # Limpar e estruturar texto
            full_text = self._clean_text(full_text)
            
            # Calcular estatísticas
            stats = {
                "total_pages": len(pages_text),
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "avg_chars_per_page": len(full_text) // max(len(pages_text), 1)
            }
            
            logger.info(
                f"PDF extraído: {filename} - "
                f"{stats['total_pages']} páginas, "
                f"{stats['total_chars']} caracteres"
            )
            
            return {
                "text": full_text,
                "pages": pages_text,
                "metadata": metadata,
                "stats": stats,
                "filename": filename
            }
        
        except Exception as e:
            logger.error(f"Erro processando PDF {filename}: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_metadata(self, reader: PdfReader) -> Dict[str, str]:
        """
        Extrai metadados do PDF
        
        Args:
            reader: Objeto PdfReader
        
        Returns:
            Dicionário com metadados
        """
        metadata = {}
        
        if reader.metadata:
            try:
                # Campos comuns de metadados
                fields = [
                    'title', 'author', 'subject', 'creator',
                    'producer', 'creation_date', 'modification_date'
                ]
                
                for field in fields:
                    value = reader.metadata.get(f'/{field.capitalize()}')
                    if value:
                        metadata[field] = str(value)
            except Exception as e:
                logger.warning(f"Erro extraindo metadados: {e}")
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Limpa e normaliza texto extraído
        
        Args:
            text: Texto bruto
        
        Returns:
            Texto limpo
        """
        # Remover múltiplas quebras de linha
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        # Remover múltiplos espaços
        import re
        text = re.sub(r' +', ' ', text)
        
        # Normalizar quebras de linha (máximo 2 consecutivas)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def summarize_for_context(
        self,
        extracted_data: Dict[str, any],
        max_chars: int = 2000
    ) -> str:
        """
        Cria resumo estruturado do PDF para inclusão no contexto
        
        Args:
            extracted_data: Dados extraídos do extract_text()
            max_chars: Tamanho máximo do resumo
        
        Returns:
            Resumo estruturado em Markdown
        """
        lines = []
        
        # Cabeçalho
        lines.append("=== DOCUMENTO PDF ===")
        lines.append(f"**Arquivo**: {extracted_data['filename']}")
        
        # Metadados (se relevantes)
        metadata = extracted_data.get('metadata', {})
        if metadata.get('title'):
            lines.append(f"**Título**: {metadata['title']}")
        if metadata.get('author'):
            lines.append(f"**Autor**: {metadata['author']}")
        
        # Estatísticas
        stats = extracted_data['stats']
        lines.append(
            f"**Páginas**: {stats['total_pages']} | "
            f"**Palavras**: {stats['total_words']}"
        )
        lines.append("")
        
        # Conteúdo
        lines.append("**CONTEÚDO**:")
        full_text = extracted_data['text']
        
        if len(full_text) <= max_chars:
            # Texto completo cabe
            lines.append(full_text)
        else:
            # Truncar: início + meio + fim
            chunk_size = (max_chars - 100) // 3
            
            start = full_text[:chunk_size]
            middle_start = len(full_text) // 2 - chunk_size // 2
            middle = full_text[middle_start:middle_start + chunk_size]
            end = full_text[-chunk_size:]
            
            lines.append(start)
            lines.append("\n... [conteúdo intermediário omitido] ...\n")
            lines.append(middle)
            lines.append("\n... [continuação] ...\n")
            lines.append(end)
        
        lines.append("")
        lines.append("=== FIM DO DOCUMENTO ===")
        
        return "\n".join(lines)
    
    def extract_structured_data(
        self,
        extracted_data: Dict[str, any],
        data_type: str = "lab_report"
    ) -> Dict[str, any]:
        """
        Tenta extrair dados estruturados de tipos específicos de documentos
        
        Args:
            extracted_data: Dados extraídos
            data_type: Tipo de documento (lab_report, radiology_report, etc)
        
        Returns:
            Dicionário com dados estruturados
        
        Note:
            Implementação básica. Em produção, usaria regex patterns
            específicos ou modelos de NER médico
        """
        text = extracted_data['text']
        
        if data_type == "lab_report":
            return self._extract_lab_values(text)
        elif data_type == "radiology_report":
            return self._extract_radiology_sections(text)
        else:
            return {"raw_text": text}
    
    def _extract_lab_values(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extrai valores de exames laboratoriais
        Implementação básica - pode ser expandida
        """
        import re
        
        # Padrão: nome_exame: valor unidade (referência)
        # Ex: "Hemoglobina: 14.5 g/dL (12-16)"
        pattern = r'([A-Za-zÀ-ÿ\s]+):\s*([0-9.,]+)\s*([a-zA-Z/%]+)?\s*(?:\(([0-9.,\-\s]+)\))?'
        
        matches = re.findall(pattern, text)
        
        lab_values = []
        for match in matches:
            test_name, value, unit, reference = match
            lab_values.append({
                "test": test_name.strip(),
                "value": value.strip(),
                "unit": unit.strip() if unit else "",
                "reference": reference.strip() if reference else ""
            })
        
        return {"lab_values": lab_values}
    
    def _extract_radiology_sections(self, text: str) -> Dict[str, str]:
        """
        Extrai seções comuns de laudos radiológicos
        """
        sections = {
            "technique": "",
            "findings": "",
            "impression": "",
            "conclusion": ""
        }
        
        # Padrões de seção (multilíngue)
        section_patterns = {
            "technique": [r'técnica:', r'technique:', r'método:'],
            "findings": [r'achados:', r'findings:', r'descrição:'],
            "impression": [r'impressão:', r'impression:', r'conclusão:'],
            "conclusion": [r'conclusão:', r'conclusion:', r'parecer:']
        }
        
        # Buscar seções (implementação básica)
        for section_key, patterns in section_patterns.items():
            for pattern in patterns:
                import re
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.end()
                    # Pegar texto até próxima seção ou fim
                    next_section = len(text)
                    for other_patterns in section_patterns.values():
                        for other_pattern in other_patterns:
                            other_match = re.search(other_pattern, text[start:], re.IGNORECASE)
                            if other_match:
                                next_section = min(next_section, start + other_match.start())
                    
                    sections[section_key] = text[start:next_section].strip()
                    break
        
        return sections


# Criar instância global
pdf_extractor = PDFExtractor()
