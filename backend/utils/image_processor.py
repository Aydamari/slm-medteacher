"""
Processador de Imagens Médicas
Redimensiona, otimiza e prepara imagens para envio ao modelo
"""

import io
import base64
import logging
from typing import Dict, Tuple, Optional
from PIL import Image, ImageEnhance

from backend.config import (
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    IMAGE_COMPRESSION_QUALITY,
    ALLOWED_IMAGE_FORMATS
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Processador de imagens médicas
    Otimiza imagens para uso com MedGemma mantendo qualidade diagnóstica
    """
    
    def __init__(self):
        self.max_file_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        self.max_dimension = MAX_IMAGE_DIMENSION
        self.compression_quality = IMAGE_COMPRESSION_QUALITY
    
    def _enhance_for_ecg(self, image: Image.Image) -> Image.Image:
        """
        Boost contrast and sharpness for ECG images with gray grids.
        Gray-grid ECGs lose readability after JPEG compression at small sizes.
        """
        image = ImageEnhance.Contrast(image).enhance(1.8)
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        return image

    def process_image(
        self,
        file_content: bytes,
        filename: str = "image.jpg",
        enhance_ecg: bool = False,
    ) -> Dict[str, any]:
        """
        Processa imagem médica para envio ao modelo
        
        Args:
            file_content: Bytes da imagem
            filename: Nome do arquivo
        
        Returns:
            Dicionário com dados processados
        """
        if len(file_content) > self.max_file_size_bytes:
            raise ValueError(
                f"Image too large: {len(file_content) / 1024 / 1024:.1f}MB. "
                f"Maximum: {MAX_FILE_SIZE_MB}MB"
            )
        
        try:
            image = Image.open(io.BytesIO(file_content))
            
            original_format = image.format
            original_size = image.size
            original_mode = image.mode
            
            if original_format.lower() not in ALLOWED_IMAGE_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {original_format}. "
                    f"Allowed: {', '.join(ALLOWED_IMAGE_FORMATS)}"
                )
            
            if image.mode not in ('RGB', 'L'):
                logger.info(f"Convertendo imagem de {image.mode} para RGB")
                image = image.convert('RGB')

            if enhance_ecg:
                image = self._enhance_for_ecg(image)
                logger.info(f"ECG contrast/sharpness enhancement aplicado: {filename}")

            image = self._resize_if_needed(image)
            optimized_bytes = self._optimize_image(image, original_format)
            base64_str = base64.b64encode(optimized_bytes).decode('utf-8')

            # #7: Use the format actually saved, not the original.
            # _optimize_image saves PNG for PNG inputs, JPEG for everything else.
            saved_format = "png" if original_format.upper() == "PNG" else "jpeg"

            final_size_kb = len(optimized_bytes) / 1024
            compression_ratio = len(file_content) / len(optimized_bytes)

            logger.info(
                f"Imagem processada: {filename} - "
                f"{original_size} → {image.size}, "
                f"{len(file_content) / 1024:.0f}KB → {final_size_kb:.0f}KB "
                f"(compressão {compression_ratio:.1f}x)"
            )

            return {
                "base64": base64_str,
                "format": saved_format,
                "size": image.size,
                "file_size_kb": round(final_size_kb, 1),
                "metadata": {
                    "original_size": original_size,
                    "original_mode": original_mode,
                    "resized": original_size != image.size,
                    "compression_ratio": round(compression_ratio, 2)
                },
                "filename": filename
            }
        
        except Exception as e:
            logger.error(f"Erro processando imagem {filename}: {e}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Redimensiona imagem se necessário"""
        width, height = image.size
        
        if width <= self.max_dimension and height <= self.max_dimension:
            return image
        
        if width > height:
            new_width = self.max_dimension
            new_height = int(height * (self.max_dimension / width))
        else:
            new_height = self.max_dimension
            new_width = int(width * (self.max_dimension / height))
        
        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        logger.info(f"Imagem redimensionada: {image.size} → {resized.size}")
        return resized
    
    def _optimize_image(self, image: Image.Image, original_format: str) -> bytes:
        """Otimiza imagem para tamanho vs qualidade"""
        buffer = io.BytesIO()
        
        if original_format.upper() == 'PNG':
            image.save(buffer, format='PNG', optimize=True)
        else:
            image.save(
                buffer,
                format='JPEG',
                quality=self.compression_quality,
                optimize=True
            )
        
        return buffer.getvalue()
    
    def create_thumbnail(
        self,
        file_content: bytes,
        max_size: Tuple[int, int] = (200, 200)
    ) -> str:
        """Cria thumbnail pequeno da imagem"""
        try:
            image = Image.open(io.BytesIO(file_content))
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=60, optimize=True)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        except Exception as e:
            logger.error(f"Erro criando thumbnail: {e}")
            return ""
    
    def summarize_for_context(
        self,
        processed_data: Dict[str, any],
        image_description: Optional[str] = None
    ) -> str:
        """Cria resumo textual da imagem para inclusão no contexto híbrido"""
        lines = []
        
        lines.append("=== IMAGEM MÉDICA ===")
        lines.append(f"**Arquivo**: {processed_data['filename']}")
        lines.append(
            f"**Dimensões**: {processed_data['size'][0]}x{processed_data['size'][1]} "
            f"| **Formato**: {processed_data['format'].upper()}"
        )
        
        if image_description:
            lines.append("")
            lines.append("**DESCRIÇÃO**:")
            lines.append(image_description)
        
        lines.append("")
        lines.append("=== FIM DA IMAGEM ===")
        
        return "\n".join(lines)


# Criar instância global
image_processor = ImageProcessor()
