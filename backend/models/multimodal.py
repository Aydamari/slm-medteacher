"""
Orquestrador Multimodal
Gerencia processamento e integração de múltiplos tipos de arquivo
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from backend.utils.pdf_extractor import pdf_extractor
from backend.utils.image_processor import image_processor
from backend.config import (
    MAX_FILES_PER_REQUEST,
    ALLOWED_IMAGE_FORMATS,
    ALLOWED_DOC_FORMATS
)

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """
    Processa múltiplos tipos de arquivo (imagens, PDFs)
    e prepara dados para envio ao modelo
    """
    
    def __init__(self):
        self.pdf_extractor = pdf_extractor
        self.image_processor = image_processor
    
    def process_files(
        self,
        files: List[Dict[str, any]],
        exam_type_hints: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Processa lista de arquivos (imagens e/ou PDFs)
        
        Args:
            files: Lista de dicionários com:
                - filename: Nome do arquivo
                - content: Bytes do arquivo
                - content_type: MIME type
        
        Returns:
            Dicionário com:
                - images: Lista de imagens processadas
                - documents: Lista de PDFs processados
                - context_summary: Resumo para inclusão no contexto híbrido
                - errors: Lista de erros (se houver)
        
        Raises:
            ValueError: Se número de arquivos excede limite
        """
        if len(files) > MAX_FILES_PER_REQUEST:
            raise ValueError(
                f"Too many files. Maximum: {MAX_FILES_PER_REQUEST} files per request. "
                f"Received: {len(files)}"
            )
        
        processed_images = []
        processed_documents = []
        errors = []
        
        for idx, file_data in enumerate(files):
            filename = file_data.get('filename', 'unknown')
            content = file_data.get('content')
            content_type = file_data.get('content_type', '')

            # ECG images need contrast/sharpness boost — gray grids lose detail at 512px
            hint = (exam_type_hints[idx] if exam_type_hints and idx < len(exam_type_hints)
                    else (exam_type_hints[0] if exam_type_hints else None))
            is_ecg_image = hint == "ecg"

            try:
                file_type = self._detect_file_type(filename, content_type)

                if file_type == 'image':
                    processed = self.image_processor.process_image(
                        content, filename, enhance_ecg=is_ecg_image
                    )
                    processed_images.append(processed)
                    logger.info(f"Imagem processada: {filename}")
                
                elif file_type == 'pdf':
                    processed = self.pdf_extractor.extract_text(content, filename)
                    processed_documents.append(processed)
                    logger.info(f"PDF processado: {filename}")
                
                else:
                    errors.append({
                        "filename": filename,
                        "error": f"Unsupported file type: {file_type}"
                    })
            
            except Exception as e:
                logger.error(f"Erro processando {filename}: {e}")
                errors.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        # Gerar resumo contextual
        context_summary = self._generate_context_summary(
            processed_images,
            processed_documents
        )
        
        return {
            "images": processed_images,
            "documents": processed_documents,
            "context_summary": context_summary,
            "errors": errors,
            "stats": {
                "total_files": len(files),
                "images_processed": len(processed_images),
                "documents_processed": len(processed_documents),
                "errors": len(errors)
            }
        }
    
    def _detect_file_type(self, filename: str, content_type: str) -> str:
        """
        Detecta tipo de arquivo baseado em extensão e MIME type
        
        Args:
            filename: Nome do arquivo
            content_type: MIME type
        
        Returns:
            'image', 'pdf', ou 'unknown'
        """
        # Por extensão
        extension = Path(filename).suffix.lower().lstrip('.')
        
        if extension in ALLOWED_IMAGE_FORMATS:
            return 'image'
        elif extension in ALLOWED_DOC_FORMATS:
            return 'pdf'
        
        # Por MIME type
        if content_type.startswith('image/'):
            return 'image'
        elif content_type == 'application/pdf':
            return 'pdf'
        
        return 'unknown'
    
    def _generate_context_summary(
        self,
        images: List[Dict],
        documents: List[Dict]
    ) -> str:
        """
        Gera resumo textual de todos os arquivos processados
        para inclusão no contexto híbrido
        
        Args:
            images: Lista de imagens processadas
            documents: Lista de PDFs processados
        
        Returns:
            String com resumo estruturado
        """
        if not images and not documents:
            return ""
        
        lines = []
        lines.append("=== ARQUIVOS ANEXADOS ===")
        lines.append("")
        
        # Resumo de imagens
        if images:
            lines.append(f"**IMAGENS ({len(images)})**:")
            for i, img in enumerate(images, 1):
                lines.append(
                    f"{i}. {img['filename']} - "
                    f"{img['size'][0]}x{img['size'][1]}px, "
                    f"{img['format'].upper()}"
                )
            lines.append("")
        
        # Resumo de documentos
        if documents:
            lines.append(f"**DOCUMENTOS ({len(documents)})**:")
            for i, doc in enumerate(documents, 1):
                stats = doc['stats']
                lines.append(
                    f"{i}. {doc['filename']} - "
                    f"{stats['total_pages']} páginas, "
                    f"{stats['total_words']} palavras"
                )
            lines.append("")
        
        lines.append("=== FIM DOS ARQUIVOS ===")
        
        return "\n".join(lines)
    
    def prepare_for_model(
        self,
        processed_data: Dict[str, any],
        user_text: str
    ) -> Tuple[str, List[str]]:
        """
        Prepara dados multimodais para envio ao modelo
        
        Args:
            processed_data: Dados do process_files()
            user_text: Texto do usuário
        
        Returns:
            Tupla (texto_completo, lista_de_imagens_base64)
        """
        # Combinar texto do usuário com conteúdo dos PDFs
        text_parts = [user_text]
        
        # Adicionar texto dos PDFs
        for doc in processed_data.get('documents', []):
            # Usar resumo em vez de texto completo (contexto híbrido)
            summary = self.pdf_extractor.summarize_for_context(doc)
            text_parts.append(summary)
        
        # Texto completo
        full_text = "\n\n".join(text_parts)
        
        # Lista de imagens em base64
        image_base64_list = [
            img['base64'] for img in processed_data.get('images', [])
        ]
        
        return full_text, image_base64_list
    
    def create_thumbnails(
        self,
        processed_data: Dict[str, any]
    ) -> List[Dict[str, str]]:
        """
        Cria thumbnails de todas as imagens para preview no frontend
        
        Args:
            processed_data: Dados do process_files()
        
        Returns:
            Lista de dicionários com {filename, thumbnail_base64}
        """
        thumbnails = []
        
        for img in processed_data.get('images', []):
            # Thumbnail já foi criado durante processamento (se implementado)
            # Aqui apenas retornamos referência
            thumbnails.append({
                "filename": img['filename'],
                "thumbnail": f"data:image/jpeg;base64,{img.get('thumbnail', img['base64'])}"
            })
        
        return thumbnails


# Criar instância global
multimodal_processor = MultimodalProcessor()
