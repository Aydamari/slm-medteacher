import logging
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Probabilidade mínima para um achado ser reportado como relevante.
# Abaixo desse valor, a patologia é descartada independente do ranking.
_MIN_FINDING_PROB = 0.15

# Máximo de achados reportados por exame
_TOP_K = 5

# Tradução dos labels TorchXRayVision → terminologia clínica.
# Labels originais vêm dos datasets de treino (NIH ChestX-ray14, CheXpert, etc.)
# e incluem termos obsoletos ou ambíguos. O mapa abaixo normaliza para português
# e adiciona o inglês entre parênteses para que o LLM reconheça sem ambiguidade.
_PATHOLOGY_LABELS_PT: Dict[str, str] = {
    "Atelectasis":              "Atelectasia",
    "Consolidation":            "Consolidação",
    "Infiltration":             "Opacidade/Infiltrado",       # termo obsoleto → equivalente atual
    "Pneumothorax":             "Pneumotórax",
    "Edema":                    "Edema Pulmonar",
    "Emphysema":                "Enfisema",
    "Fibrosis":                 "Fibrose Pulmonar",
    "Effusion":                 "Derrame Pleural",
    "Pneumonia":                "Pneumonia",
    "Pleural_Thickening":       "Espessamento Pleural",
    "Cardiomegaly":             "Cardiomegalia",
    "Nodule":                   "Nódulo Pulmonar",
    "Mass":                     "Massa Pulmonar",
    "Hernia":                   "Hérnia Diafragmática",
    "Lung Lesion":              "Lesão Pulmonar",
    "Fracture":                 "Fratura Óssea",
    "Lung Opacity":             "Opacidade Pulmonar",
    "Enlarged Cardiomediastinum": "Alargamento do Mediastino",
}


class XRayAnalyzer:
    """
    Analisa radiografias de tórax usando ensemble de modelos TorchXRayVision.

    Degradação graceful: se apenas um modelo carregar, o ensemble reduz para
    single-model em vez de falhar completamente.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dense: Optional[xrv.models.DenseNet] = None
        self.model_res: Optional[xrv.models.ResNet] = None
        self.model_nih: Optional[xrv.models.DenseNet] = None
        self.model_chex: Optional[xrv.models.DenseNet] = None

        logger.info("Inicializando Pipeline de Raio-X...")

        # Modelo 1: DenseNet121 (224×224) — try/except independente
        try:
            self.model_dense = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model_dense.to(self.device).eval()
            logger.info("DenseNet121 (res224-all) carregado.")
        except Exception as e:
            logger.error(f"DenseNet121 não carregado: {e}")

        # Modelo 2: ResNet50 (512×512) — try/except independente
        try:
            self.model_res = xrv.models.ResNet(weights="resnet50-res512-all")
            self.model_res.to(self.device).eval()
            logger.info("ResNet50 (res512-all) carregado.")
        except Exception as e:
            logger.error(f"ResNet50 não carregado: {e}")

        # Modelo 3: DenseNet121 NIH — tem "Emphysema" como label explícito
        try:
            self.model_nih = xrv.models.DenseNet(weights="densenet121-res224-nih")
            self.model_nih.to(self.device).eval()
            logger.info("DenseNet121 (NIH) carregado.")
        except Exception as e:
            logger.error(f"DenseNet121-NIH não carregado: {e}")

        # Modelo 4: DenseNet121 CheXpert — melhor para consolidação/infiltrado
        try:
            self.model_chex = xrv.models.DenseNet(weights="densenet121-res224-chex")
            self.model_chex.to(self.device).eval()
            logger.info("DenseNet121 (CheXpert) carregado.")
        except Exception as e:
            logger.error(f"DenseNet121-CheXpert não carregado: {e}")

        available = sum([
            self.model_dense is not None,
            self.model_res is not None,
            self.model_nih is not None,
            self.model_chex is not None,
        ])
        if available == 0:
            logger.error("Nenhum modelo de RX carregado. Pipeline indisponível.")
        elif available == 1:
            names = [
                name for name, m in [
                    ("DenseNet121", self.model_dense), ("ResNet50", self.model_res),
                    ("DenseNet121-NIH", self.model_nih), ("DenseNet121-CheX", self.model_chex),
                ] if m is not None
            ]
            logger.warning(f"Pipeline RX em modo single-model ({names[0]}).")
        else:
            names = [
                name for name, m in [
                    ("DenseNet121", self.model_dense), ("ResNet50", self.model_res),
                    ("DenseNet121-NIH", self.model_nih), ("DenseNet121-CheX", self.model_chex),
                ] if m is not None
            ]
            logger.info(f"Ensemble RX ({' + '.join(names)}) pronto.")

    def _pathologies(self) -> List[str]:
        """Retorna a lista de patologias do modelo de referência disponível."""
        ref = self.model_dense or self.model_res or self.model_nih or self.model_chex
        return list(ref.pathologies) if ref else []

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """Classifica patologias em radiografia de tórax.

        Usa ensemble de até 4 modelos com conjuntos de labels distintos.
        A média é feita por label (dict merge), não por tensor, para acomodar
        diferentes números de saídas (NIH=14, CheXpert=14, all=18).

        Returns:
            dict com "status", "findings", "method" em sucesso;
            dict com "error" em falha.
        """
        if self.model_dense is None and self.model_res is None and self.model_nih is None and self.model_chex is None:
            return {"error": "Nenhum modelo de RX disponível."}

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                return {"error": "Falha ao decodificar imagem."}

            # Normalização TorchXRayVision: [0, 255] → [-1024, 1024]
            # .astype(np.float32) garante dtype correto independente da versão do TXV
            img_norm = xrv.datasets.normalize(img_gray, 255).astype(np.float32)

            # Cada modelo retorna scores para seu próprio conjunto de labels.
            # Acumulamos como dict {label: [score, ...]} para média por label.
            label_scores: Dict[str, List[float]] = {}
            active_models: List[str] = []

            def _accumulate(model_out: np.ndarray, pathologies: List[str]) -> None:
                for label, score in zip(pathologies, model_out):
                    if not label:
                        continue
                    if label not in label_scores:
                        label_scores[label] = []
                    label_scores[label].append(float(score))

            with torch.no_grad():
                if self.model_dense is not None:
                    img_224 = cv2.resize(img_norm, (224, 224))
                    inp = torch.from_numpy(img_224[None, :, :]).unsqueeze(0).to(self.device)
                    out = self.model_dense(inp)[0].cpu().numpy()
                    _accumulate(out, list(self.model_dense.pathologies))
                    active_models.append("DenseNet121")

                if self.model_res is not None:
                    img_512 = cv2.resize(img_norm, (512, 512))
                    inp = torch.from_numpy(img_512[None, :, :]).unsqueeze(0).to(self.device)
                    out = self.model_res(inp)[0].cpu().numpy()
                    _accumulate(out, list(self.model_res.pathologies))
                    active_models.append("ResNet50")

                if self.model_nih is not None:
                    img_224 = cv2.resize(img_norm, (224, 224))
                    inp = torch.from_numpy(img_224[None, :, :]).unsqueeze(0).to(self.device)
                    out = self.model_nih(inp)[0].cpu().numpy()
                    _accumulate(out, list(self.model_nih.pathologies))
                    active_models.append("DenseNet121-NIH")

                if self.model_chex is not None:
                    img_224 = cv2.resize(img_norm, (224, 224))
                    inp = torch.from_numpy(img_224[None, :, :]).unsqueeze(0).to(self.device)
                    out = self.model_chex(inp)[0].cpu().numpy()
                    _accumulate(out, list(self.model_chex.pathologies))
                    active_models.append("DenseNet121-CheX")

            # Média por label (labels únicos de cada modelo contribuem com 1 voto)
            avg_scores: Dict[str, float] = {
                label: float(np.mean(scores)) for label, scores in label_scores.items()
            }

            results: List[Tuple[str, float]] = sorted(
                avg_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Ordenar por probabilidade e aplicar threshold mínimo
            above_threshold = [(p, prob) for p, prob in results if prob >= _MIN_FINDING_PROB]

            # Garantir ao menos 1 achado (mesmo abaixo do threshold) para não
            # deixar o LLM sem dado de referência
            top_k = above_threshold[:_TOP_K] if above_threshold else results[:1]

            method = f"Ensemble ({' + '.join(active_models)})" if len(active_models) > 1 else active_models[0]

            return {
                "status": "success",
                "findings": top_k,       # lista de (label, prob) — formatada em get_summary
                "method": method,
                "models_active": active_models,
                "findings_below_threshold": len(results) - len(above_threshold),
            }

        except Exception as e:
            logger.error(f"Erro na análise de Raio-X: {e}")
            return {"error": str(e)}

    def get_summary_for_prompt(
        self, analysis_results: Dict[str, Any], language: str = "en"
    ) -> str:
        """Retorna bloco de DADOS para o prompt — sem template de formato.

        O template de formato (estrutura do laudo) fica em exam_instructions.py,
        seguindo o mesmo padrão do ECGAnalyzer. Aqui apenas estruturamos os dados
        da IA para que o LLM os utilize como suporte à análise visual.
        """
        if "error" in analysis_results:
            if language == "pt":
                return (
                    "\n[ANÁLISE AUTOMÁTICA (IA) INDISPONÍVEL]\n"
                    "Prossiga com análise visual direta da imagem.\n"
                    "---\n"
                )
            return (
                "\n[AUTOMATED AI ANALYSIS UNAVAILABLE]\n"
                "Proceed with direct visual analysis of the image.\n"
                "---\n"
            )

        findings: List[Tuple[str, float]] = analysis_results.get("findings", [])
        method = analysis_results.get("method", "IA")
        below = analysis_results.get("findings_below_threshold", 0)

        if not findings:
            if language == "pt":
                return "\n[DADOS DE SUPORTE (IA)]: Nenhum achado acima do limiar de confiança.\n---\n"
            return "\n[SUPPORTING DATA (AI)]: No findings above confidence threshold.\n---\n"

        # Formatar achados com labels traduzidos (PT) ou originais (EN)
        def _fmt(label: str, prob: float) -> str:
            if language == "pt":
                translated = _PATHOLOGY_LABELS_PT.get(label, label)
                return f"- {translated} ({label}): {prob:.0%}"
            return f"- {label}: {prob:.0%}"

        findings_lines = "\n".join(_fmt(p, prob) for p, prob in findings)

        if language == "pt":
            note = (
                f"\n  *(+ {below} achado(s) abaixo do limiar de confiança de {_MIN_FINDING_PROB:.0%} omitidos)*"
                if below > 0 else ""
            )
            return (
                f"\n[DADOS DE SUPORTE — IA ({method}) — Top {len(findings)} por probabilidade]:\n"
                f"{findings_lines}{note}\n"
                "---\n"
            )
        note = (
            f"\n  *({below} finding(s) below confidence threshold of {_MIN_FINDING_PROB:.0%} omitted)*"
            if below > 0 else ""
        )
        return (
            f"\n[SUPPORTING DATA — AI ({method}) — Top {len(findings)} by probability]:\n"
            f"{findings_lines}{note}\n"
            "---\n"
        )


# --- Lazy factory (padrão dos demais analyzers) ---

_xray_analyzer: Optional[XRayAnalyzer] = None
_xray_analyzer_error: Optional[str] = None


def get_xray_analyzer() -> XRayAnalyzer:
    """Lazy factory — instancia XRayAnalyzer na primeira chamada."""
    global _xray_analyzer, _xray_analyzer_error
    if _xray_analyzer is None and _xray_analyzer_error is None:
        try:
            _xray_analyzer = XRayAnalyzer()
            # Falha parcial: se nenhum modelo carregou, marcar como erro
            if (
                _xray_analyzer.model_dense is None
                and _xray_analyzer.model_res is None
                and _xray_analyzer.model_nih is None
                and _xray_analyzer.model_chex is None
            ):
                _xray_analyzer_error = "Nenhum modelo de RX disponível."
                _xray_analyzer = None
        except Exception as e:
            _xray_analyzer_error = str(e)
            logger.exception(f"Falha ao inicializar XRayAnalyzer: {e}")
    if _xray_analyzer_error is not None and _xray_analyzer is None:
        raise RuntimeError(f"XRayAnalyzer init failed: {_xray_analyzer_error}")
    return _xray_analyzer


def is_xray_analyzer_loaded() -> bool:
    return _xray_analyzer is not None


def xray_analyzer_error() -> Optional[str]:
    return _xray_analyzer_error
