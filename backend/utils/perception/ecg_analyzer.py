import logging
import os
import torch
import numpy as np
import cv2
import yaml
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def ensure_dependencies():
    """Verifica e instala dependências críticas automaticamente."""
    deps = {
        "tensorflow": "tensorflow>=2.16.0",
        "nnunetv2": "nnunetv2"
    }
    for mod, pkg in deps.items():
        if importlib.util.find_spec(mod) is None:
            logger.info(f"Instalando dependência ausente: {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                logger.info(f"{pkg} instalado com sucesso.")
            except Exception as e:
                logger.error(f"Falha ao instalar {pkg}: {e}")

ensure_dependencies()

# Configuração de caminhos
from yacs.config import CfgNode as CN
from torchvision import transforms
from PIL import Image
import timm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=500.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Configuração de caminhos
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
AHUS_AIM_DIR = PROJECT_ROOT / "backend" / "utils" / "perception" / "ahus_aim"
if str(AHUS_AIM_DIR) not in sys.path: sys.path.append(str(AHUS_AIM_DIR))

try:
    from src.model.inference_wrapper import InferenceWrapper
    logger.info("InferenceWrapper importado com sucesso.")
except ImportError as e:
    InferenceWrapper = None
    logger.warning(f"InferenceWrapper indisponível: {e}")

# Prioridade clínica para desempate no consenso (maior = mais grave)
CLINICAL_SEVERITY = {
    'AF': 6, 'LBBB': 5, 'RBBB': 4, '1dAVb': 3, 'ST': 2, 'SB': 1,
    'MI': 6, 'HYP': 4, 'CD': 3, 'STTC': 2, 'NORM': 0,
    'Sinusal/Normal': 0, 'Indeterminado': -1
}

# Correlações clínicas obrigatórias por diagnóstico.
# Injetadas no bloco de dados ANTES do LLM ver o template de resposta,
# para evitar que o modelo descreva achados incompatíveis com a HD detectada.
_CLINICAL_CORRELATIONS: Dict[str, Dict[str, str]] = {
    "AF": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — FA DETECTADA:\n"
            "  • Onda P: AUSENTE — substituída por atividade fibrilatória irregular (ondas f de baixa amplitude)\n"
            "  • Ritmo: IRREGULARMENTE IRREGULAR (intervalos RR variáveis sem padrão)\n"
            "  • Intervalo PR: INDEFINIDO (sem onda P discernível)\n"
            "  IMPORTANTE: NÃO descreva ondas P presentes — achado incompatível com FA."
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — AF DETECTED:\n"
            "  • P Wave: ABSENT — replaced by irregular fibrillatory activity (low-amplitude f waves)\n"
            "  • Rhythm: IRREGULARLY IRREGULAR (variable RR intervals without pattern)\n"
            "  • PR Interval: UNDEFINED (no discernible P wave)\n"
            "  IMPORTANT: DO NOT describe P waves as present — incompatible with AF."
        ),
    },
    "LBBB": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — BRE DETECTADO:\n"
            "  • QRS: ALARGADO (> 120 ms) — morfologia em M (RsR') em V5/V6; QS ou rS em V1\n"
            "  • Onda T: DISCORDANTE ao QRS (inversão em derivações com QRS positivo)\n"
            "  • Segmento ST: Discordante — não interpretar como sinal isolado de isquemia\n"
            "  • Critério de Sgarbossa necessário para diagnosticar IAM em presença de BRE."
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — LBBB DETECTED:\n"
            "  • QRS: WIDE (> 120 ms) — M-shape (RsR') in V5/V6; QS or rS in V1\n"
            "  • T Wave: DISCORDANT to QRS (inversion in leads with positive QRS)\n"
            "  • ST segment: Discordant — do not interpret as isolated ischemia\n"
            "  • Sgarbossa criteria required to diagnose AMI in the presence of LBBB."
        ),
    },
    "RBBB": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — BRD DETECTADO:\n"
            "  • QRS: ALARGADO (> 120 ms) — padrão RSR' (orelha de coelho) em V1/V2; onda S larga em D1, aVL, V5/V6\n"
            "  • Onda T: Discordante nas derivações direitas (V1–V3)"
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — RBBB DETECTED:\n"
            "  • QRS: WIDE (> 120 ms) — RSR' pattern (rabbit ears) in V1/V2; broad S wave in I, aVL, V5/V6\n"
            "  • T Wave: Discordant in right precordial leads (V1–V3)"
        ),
    },
    "1dAVb": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — BAV 1º GRAU:\n"
            "  • Intervalo PR: PROLONGADO (> 200 ms / > 5 quadradinhos)\n"
            "  • Onda P: Presente e precedendo cada QRS (condução AV mantida, apenas lentificada)"
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — 1st DEGREE AV BLOCK:\n"
            "  • PR Interval: PROLONGED (> 200 ms / > 5 small squares)\n"
            "  • P Wave: Present and preceding each QRS (AV conduction maintained, only slowed)"
        ),
    },
    "SB": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — BRADICARDIA SINUSAL:\n"
            "  • FC: < 60 bpm\n"
            "  • Ritmo: Regular, sinusal (onda P presente, morfologia normal, PR normal)"
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — SINUS BRADYCARDIA:\n"
            "  • HR: < 60 bpm\n"
            "  • Rhythm: Regular, sinus (P wave present, normal morphology, normal PR)"
        ),
    },
    "ST": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — TAQUICARDIA SINUSAL:\n"
            "  • FC: > 100 bpm\n"
            "  • Ritmo: Regular, sinusal (onda P presente antes de cada QRS, PR normal)"
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — SINUS TACHYCARDIA:\n"
            "  • HR: > 100 bpm\n"
            "  • Rhythm: Regular, sinus (P wave present before each QRS, normal PR)"
        ),
    },
    "MI": {
        "pt": (
            "⚠ CORRELAÇÃO CLÍNICA — INFARTO DO MIOCÁRDIO (PTB-XL):\n"
            "  • Pesquise: supradesnivelamento de ST (IAMCSST), onda Q patológica (Q ≥ 1 mm largura/25% amplitude QRS),\n"
            "    infradesnivelamento de ST, inversão de onda T\n"
            "  • Localize o território afetado pela distribuição das derivações alteradas"
        ),
        "en": (
            "⚠ CLINICAL CORRELATION — MYOCARDIAL INFARCTION (PTB-XL):\n"
            "  • Look for: ST elevation (STEMI), pathological Q wave (≥ 1 mm wide / 25% QRS amplitude),\n"
            "    ST depression, T wave inversion\n"
            "  • Localize affected territory by distribution of abnormal leads"
        ),
    },
}


def _get_clinical_note(diagnosis: str, language: str) -> str:
    """Retorna nota clínica para o diagnóstico detectado, ou string vazia."""
    entry = _CLINICAL_CORRELATIONS.get(diagnosis, {})
    return entry.get(language, entry.get("en", ""))


class ECGAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)

        self.digitizer = None
        self.bhf_model = None
        self.queenbee = None

        self.bhf_classes = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
        self.ptbxl_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

        # 1. BHF ConvNeXt — DESABILITADO DO CONSENSO
        # A cabeça classificadora (head.fc) está ausente dos pesos pretrained do timm
        # (confirmed: "Missing keys: head.fc.weight, head.fc.bias") — ou seja, é aleatória.
        # Um classificador com pesos aleatórios NÃO pode contribuir para diagnóstico clínico.
        # Permanece None até que pesos de fine-tuning cardíaco sejam obtidos.
        self.bhf_model = None
        logger.info("BHF ConvNeXt: desabilitado (cabeça classificadora sem fine-tuning cardíaco).")

        # 3. Carregar Queenbee via snapshot_download + importação manual de model.py
        # AutoModel.from_pretrained não funciona porque o config.json não tem auto_map.
        # O modelo exige instanciação manual com ECGTransformer + torch.load do checkpoint.
        try:
            logger.info("Tentando carregar Queenbee-ECG Transformer...")
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download("Trustcat/queenbee-ecg-transformer")
            spec = importlib.util.spec_from_file_location(
                "queenbee_model", f"{model_dir}/model.py"
            )
            queenbee_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(queenbee_mod)
            _ECGTransformer = queenbee_mod.ECGTransformer
            qb_model = _ECGTransformer(
                num_leads=12,
                signal_length=5000,
                patch_size=50,
                embed_dim=256,
                depth=6,
                num_heads=8,
                num_superclasses=5,
                num_scp_codes=44,
            )
            # weights_only=False: PyTorch 2.6 changed the default to True, but this
            # checkpoint requires numpy globals (numpy._core.multiarray.scalar).
            # Safe because the file comes from a known HuggingFace repo.
            ckpt = torch.load(
                f"{model_dir}/ecg_transformer_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            qb_model.load_state_dict(ckpt["model_state_dict"])
            self.queenbee = qb_model.to(self.torch_device).eval()
            logger.info("Queenbee-ECG carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro no Queenbee: {e}")

        # 3.4. Carregar ResNet CODE-15 (Modelo Clínico Principal - USP/PhysioNet)
        self.code15_model = None
        try:
            import tensorflow as tf
            # Forçar CPU para CODE-15: modelo pequeno (25 MB); evita conflitos entre
            # TF e PyTorch pelo mesmo contexto CUDA e erros de libdevice/ptxas.
            tf.config.set_visible_devices([], 'GPU')
            model_path = str(PROJECT_ROOT / 'backend/models/resnet_code15.h5')
            if Path(model_path).exists():
                # compile=False: modelo salvo sem configuração de treino (normal para
                # checkpoints de inferência). Funciona perfeitamente para predict().
                self.code15_model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Modelo ResNet CODE-15 carregado com sucesso (CPU, sem compilação).")
            else:
                logger.warning(f"ResNet CODE-15 não encontrado em {model_path}")
        except Exception as e:
            logger.warning(f"Falha ao carregar ResNet CODE-15: {e}. Certifique-se de que 'tensorflow' está instalado.")

        # 3.5. Carregar ECGFounder (PKUDigitalHealth/ECGFounder — MIT)
        # net1d.py não está incluído no snapshot do HuggingFace Hub — está disponível
        # apenas no repositório GitHub. Por isso é mantido em backend/models/weights/ecgfounder/
        # e baixado uma vez via wget (ver documentação em docs/pipeline_completo.md).
        self.ecgfounder_model = None
        self.ecgfounder_labels: List[str] = []
        try:
            from huggingface_hub import snapshot_download
            ecgf_dir = snapshot_download("PKUDigitalHealth/ECGFounder")

            # net1d.py: preferir cópia local controlada; HF snapshot não inclui código fonte
            local_net1d = PROJECT_ROOT / "backend" / "models" / "weights" / "ecgfounder" / "net1d.py"
            if local_net1d.exists():
                net1d_path = str(local_net1d)
            else:
                logger.warning("ECGFounder: net1d.py não encontrado em backend/models/weights/ecgfounder/. "
                               "Baixe com: wget -O backend/models/weights/ecgfounder/net1d.py "
                               "https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/main/net1d.py")
                raise FileNotFoundError("net1d.py não encontrado")

            ecgf_spec = importlib.util.spec_from_file_location("ecgfounder_net1d", net1d_path)
            ecgf_mod = importlib.util.module_from_spec(ecgf_spec)
            ecgf_spec.loader.exec_module(ecgf_mod)
            Net1D = ecgf_mod.Net1D
            ecgf_model = Net1D(
                in_channels=12,
                base_filters=64,
                ratio=1,
                filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                kernel_size=16,
                stride=2,
                groups_width=16,
                verbose=False,
                use_bn=False,
                use_do=False,
                n_classes=150,
            )
            ckpt_path = Path(ecgf_dir) / "12_lead_ECGFounder.pth"
            ecgf_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ecgf_model.load_state_dict(ecgf_ckpt["state_dict"], strict=False)
            self.ecgfounder_model = ecgf_model.to(self.torch_device).eval()

            # tasks.txt não está no snapshot do HuggingFace Hub — usar cópia local
            local_tasks = PROJECT_ROOT / "backend" / "models" / "weights" / "ecgfounder" / "tasks.txt"
            hf_tasks    = Path(ecgf_dir) / "tasks.txt"
            tasks_path  = local_tasks if local_tasks.exists() else hf_tasks
            if tasks_path.exists():
                with open(tasks_path) as f:
                    self.ecgfounder_labels = [line.strip() for line in f if line.strip()]
            logger.info(f"ECGFounder carregado ({len(self.ecgfounder_labels)} labels).")
        except Exception as e:
            logger.warning(f"ECGFounder não disponível: {e}")

        # 4. Inicializar Digitizer (Ahus-AIM)
        if InferenceWrapper:
            try:
                self._init_digitizer()
                if self.digitizer is not None:
                    logger.info("Digitizer Ahus-AIM inicializado com sucesso.")
                else:
                    logger.warning("Digitizer Ahus-AIM indisponível (pesos não materializados).")
            except Exception as e:
                logger.exception(f"Erro ao configurar Digitizer: {e}")
        else:
            logger.warning("Digitizer indisponível: InferenceWrapper não foi importado.")

    @staticmethod
    def _validate_weight_file(path: str, label: str) -> bool:
        """#2: Validate that a .pt weight file is not a Git LFS pointer."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"{label}: weight file not found at {path}")
            return False
        size = p.stat().st_size
        if size < 1024:  # LFS pointers are ~130 bytes
            logger.error(
                f"{label}: weight file at {path} is only {size} bytes — "
                "likely a Git LFS pointer. Run 'git lfs pull' to materialise weights."
            )
            return False
        return True

    def _init_digitizer(self):
        # Validate weight files before loading (#2)
        seg_path = str(PROJECT_ROOT / 'backend/models/ecg_segmentation_unet.pt')
        lead_path = str(PROJECT_ROOT / 'backend/models/lead_name_unet_weights_07072025.pt')
        if not self._validate_weight_file(seg_path, "ECG Segmentation UNet"):
            logger.warning("Digitizer disabled: segmentation weights not materialised.")
            return
        if not self._validate_weight_file(lead_path, "Lead Name UNet"):
            logger.warning("Digitizer disabled: lead-name weights not materialised.")
            return

        cfg = CN()
        cfg.SIGNAL_EXTRACTOR = CN(); cfg.SIGNAL_EXTRACTOR.class_path = 'src.model.signal_extractor.SignalExtractor'; cfg.SIGNAL_EXTRACTOR.KWARGS = CN()
        cfg.PERSPECTIVE_DETECTOR = CN(); cfg.PERSPECTIVE_DETECTOR.class_path = 'src.model.perspective_detector.PerspectiveDetector'; cfg.PERSPECTIVE_DETECTOR.KWARGS = CN(); cfg.PERSPECTIVE_DETECTOR.KWARGS.num_thetas = 250
        cfg.DEWARPER = CN(); cfg.DEWARPER.class_path = 'src.model.dewarper.Dewarper'; cfg.DEWARPER.KWARGS = CN(); cfg.DEWARPER.KWARGS.abs_peak_threshold = 0.1
        cfg.SEGMENTATION_MODEL = CN(); cfg.SEGMENTATION_MODEL.class_path = 'src.model.unet.UNet'; cfg.SEGMENTATION_MODEL.weight_path = str(PROJECT_ROOT / 'backend/models/ecg_segmentation_unet.pt'); cfg.SEGMENTATION_MODEL.KWARGS = CN(); cfg.SEGMENTATION_MODEL.KWARGS.num_in_channels = 3; cfg.SEGMENTATION_MODEL.KWARGS.num_out_channels = 4; cfg.SEGMENTATION_MODEL.KWARGS.dims = [32, 64, 128, 256, 320, 320, 320, 320]; cfg.SEGMENTATION_MODEL.KWARGS.depth = 2
        cfg.CROPPER = CN(); cfg.CROPPER.class_path = 'src.model.cropper.Cropper'; cfg.CROPPER.KWARGS = CN(); cfg.CROPPER.KWARGS.granularity = 80; cfg.CROPPER.KWARGS.percentiles = [0.02, 0.98]; cfg.CROPPER.KWARGS.alpha = 0.85
        cfg.PIXEL_SIZE_FINDER = CN(); cfg.PIXEL_SIZE_FINDER.class_path = 'src.model.pixel_size_finder.PixelSizeFinder'; cfg.PIXEL_SIZE_FINDER.KWARGS = CN(); cfg.PIXEL_SIZE_FINDER.KWARGS.min_number_of_grid_lines = 30; cfg.PIXEL_SIZE_FINDER.KWARGS.max_number_of_grid_lines = 70; cfg.PIXEL_SIZE_FINDER.KWARGS.lower_grid_line_factor = 0.3
        # E2: Usar lead_layouts_all.yml (13 layouts) em vez de _reduced.yml (2 layouts)
        cfg.LAYOUT_IDENTIFIER = CN(); cfg.LAYOUT_IDENTIFIER.class_path = 'src.model.lead_identifier.LeadIdentifier'; cfg.LAYOUT_IDENTIFIER.config_path = str(AHUS_AIM_DIR / 'src/config/lead_layouts_all.yml'); cfg.LAYOUT_IDENTIFIER.unet_config_path = str(AHUS_AIM_DIR / 'src/config/lead_name_unet.yml'); cfg.LAYOUT_IDENTIFIER.unet_weight_path = str(PROJECT_ROOT / 'backend/models/lead_name_unet_weights_07072025.pt'); cfg.LAYOUT_IDENTIFIER.KWARGS = CN(); cfg.LAYOUT_IDENTIFIER.KWARGS.debug = False; cfg.LAYOUT_IDENTIFIER.KWARGS.device = self.device; cfg.LAYOUT_IDENTIFIER.KWARGS.possibly_flipped = False

        self.digitizer = InferenceWrapper(config=cfg, device=self.device, apply_dewarping=True)

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """Pipeline ECG com prioridade SOTA e fallback Ahus-AIM:

        Nível 1 (SOTA):       ECG-Digitiser (nnU-Net + Hough) -> CODE-15 + Queenbee
        Nível 2 (Success):    Ahus-AIM (U-Net) -> CODE-15 + Queenbee
        Nível 3 (Partial):    Visual-Only fallback
        """
        try:
            warnings = []
            signal_np = None
            method_used = ""
            nk_results = None
            rhythm_strip = None

            # 1. Tentar Digitalizador SOTA (Felix Krones - PhysioNet 2024)
            try:
                from .ecg_digitiser_sota import get_sota_digitiser
                sota_engine = get_sota_digitiser(device=self.device)
                signal_np = sota_engine.digitize(image_bytes)
                if signal_np is not None:
                    method_used = "SOTA (nnU-Net + Hough)"
                    logger.info("Sinal extraído via motor SOTA.")
            except Exception as se:
                logger.debug(f"Motor SOTA indisponível ou falhou: {se}")

            # 2. Fallback para Ahus-AIM (Digitalizador Base)
            if signal_np is None:
                signal_np = self._extract_signal(image_bytes)
                if signal_np is not None:
                    method_used = "Ahus-AIM (Optimized PC)"
                    logger.info("Sinal extraído via motor Ahus-AIM.")

            # 3. Fallback OpenCV rhythm strip (quando ambos os digitalizadores falham)
            if signal_np is None:
                rhythm_strip = self._extract_rhythm_strip(image_bytes)
                if rhythm_strip is not None:
                    logger.info("Tira de ritmo extraída via OpenCV (fallback).")

            # 4. Análise rule-based neurokit2 com Lead II (quando sinal digital disponível)
            if signal_np is not None:
                lead_ii = self._get_lead_ii(signal_np)
                if lead_ii is not None:
                    nk_results = self._analyze_with_neurokit2(lead_ii, sample_rate=500)
            elif rhythm_strip is not None:
                nk_results = self._analyze_with_neurokit2(
                    np.array(rhythm_strip["signal"], dtype=np.float32),
                    sample_rate=rhythm_strip.get("sample_rate", 500),
                )

            # 5. Modelos de sinal (inferência baseada no sinal extraído)
            res_hubert = None
            res_queenbee = None
            res_code15 = None

            if signal_np is not None:
                res_hubert = self._predict_hubert(signal_np)
                res_queenbee = self._predict_queenbee(signal_np) if self.queenbee else None
                res_code15 = self._predict_code15(signal_np) if self.code15_model else None
            elif self.digitizer is None and rhythm_strip is None:
                warnings.append("Digitalização falhou: nenhum digitalizador disponível.")

            # 6. Consenso (CODE-15 > Queenbee > ECGFounder)
            # BHF ConvNeXt excluído: cabeça classificadora sem fine-tuning cardíaco.
            votes = [r for r in [res_code15, res_queenbee, res_hubert] if r is not None and r.get("label") != "Error"]
            final_diagnosis = self._consensus_vote(votes)

            # 7. Frequência cardíaca: priorizar neurokit2 (mais robusto que heurística)
            if nk_results and nk_results.get("hr"):
                hr = nk_results["hr"]
            elif signal_np is not None:
                hr = self._estimate_hr(signal_np)
            else:
                hr = None

            # 8. Determinar status
            if signal_np is not None:
                status = "success"
            elif rhythm_strip is not None:
                status = "partial_cv"
            elif votes:
                status = "partial"
            else:
                status = "error"

            active_models = []
            if res_code15 is not None:   active_models.append("CODE-15")
            if res_queenbee is not None: active_models.append("Queenbee")
            if res_hubert is not None:   active_models.append("ECGFounder")
            if nk_results is not None:   active_models.append("neurokit2")

            if signal_np is not None:
                method = f"Ensemble ({', '.join(active_models)})" if active_models else "Digitalização sem classificador"
            elif rhythm_strip is not None:
                method = "OpenCV Strip + neurokit2"
            else:
                method = "Visual-Only (sem digitalizador)"

            # Achados base dos modelos de voto
            findings_list = [f"{r['label']} ({r['prob']:.2f})" for r in votes]

            # ECGFounder top-5: adicionados como achados separados para o LLM
            if res_hubert and res_hubert.get("all_probs"):
                _ECG_FINDING_THRESHOLD = 0.15
                sorted_ecgf = sorted(
                    res_hubert["all_probs"].items(), key=lambda x: x[1], reverse=True
                )
                ecgf_top5 = [
                    f"[ECGFounder] {label} ({prob:.2f})"
                    for label, prob in sorted_ecgf[:5]
                    if prob >= _ECG_FINDING_THRESHOLD
                ]
                findings_list.extend(ecgf_top5)

            result = {
                "status": status,
                "rhythm": final_diagnosis,
                "hr": hr,
                "findings": findings_list,
                "method": method,
            }

            if nk_results:
                result["nk_rhythm"] = nk_results.get("rhythm")
                result["nk_qrs_duration_ms"] = nk_results.get("qrs_duration_ms")
                result["nk_rr_cv"] = nk_results.get("rr_cv")

            if votes:
                result["max_prob"] = max(r["prob"] for r in votes)
            if warnings:
                result["warnings"] = warnings

            return result

        except Exception as e:
            logger.exception(f"Erro na pipeline ECG: {e}")
            return {"error": str(e)}

    def _predict_bhf(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            output = self.bhf_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs).item()
            return {"label": self.bhf_classes[idx], "prob": float(probs[0][idx])}

    def _predict_code15(self, signal_np) -> Optional[Dict[str, Any]]:
        """Inference using the ResNet CODE-15 model (Keras .h5).
        
        Input: signal_np (12, 5000) @ 500 Hz.
        Output: Top diagnosis label and probability.
        """
        if self.code15_model is None:
            return None
        try:
            # CODE-15 ResNet expects (batch, samples, leads) = (1, 4096, 12)
            # Our signal is (12, 5000). We need to resample and transpose.
            sig = signal_np
            if sig.ndim == 3: sig = sig[0] # remove batch if present
            
            # Resampling to 4096 samples (CODE-15 architecture)
            if sig.shape[-1] != 4096:
                x = np.linspace(0, 1, sig.shape[-1])
                x_new = np.linspace(0, 1, 4096)
                sig = interp1d(x, sig, axis=-1)(x_new)
            
            # Transpose to (4096, 12) and add batch dim -> (1, 4096, 12)
            input_data = sig.T[np.newaxis, ...]
            
            # Predict
            probs = self.code15_model.predict(input_data, verbose=0)[0]
            
            # Labels CODE-15 (Standard 6 classes)
            # Normal, AF, LBBB, RBBB, 1dAVb, ST
            labels = ['NORM', 'AF', 'LBBB', 'RBBB', '1dAVb', 'ST']
            top_idx = np.argmax(probs)
            
            return {
                "label": labels[top_idx] if top_idx < len(labels) else "Other",
                "prob": float(probs[top_idx])
            }
        except Exception as e:
            logger.warning(f"CODE-15 inference failed: {e}")
            return None

    def _predict_hubert(self, signal_np) -> Optional[Dict[str, Any]]:
        """ECGFounder inference — 150 diagnósticos via Net1D (MIT License).

        Input: signal_np (12, 5000) ou (1, 12, 5000) @ 500 Hz.
        Output: {"label": str, "prob": float, "all_probs": {label: prob}}.
        Retorna None se ECGFounder indisponível ou em caso de erro.
        """
        if self.ecgfounder_model is None:
            return None
        try:
            sig = signal_np
            # Normalizar shape: garantir (12, 5000)
            if sig.ndim == 3:
                sig = sig[0]  # (1, 12, 5000) → (12, 5000)
            # Resampling se necessário (ECGFounder espera exatamente 5000 amostras)
            if sig.shape[-1] != 5000:
                x = np.linspace(0, 1, sig.shape[-1])
                x_new = np.linspace(0, 1, 5000)
                sig = interp1d(x, sig, axis=-1)(x_new)
            # NaN handling + z-score normalization por canal (lead)
            sig = np.nan_to_num(sig.astype(np.float32), nan=0.0)
            mean = sig.mean(axis=-1, keepdims=True)
            std = sig.std(axis=-1, keepdims=True)
            sig = (sig - mean) / (std + 1e-8)
            # Tensor (1, 12, 5000)
            input_tensor = torch.from_numpy(sig).unsqueeze(0).float().to(self.torch_device)
            with torch.no_grad():
                logits = self.ecgfounder_model(input_tensor)  # (1, 150)
                probs = torch.sigmoid(logits)[0].cpu().numpy()  # (150,)
            top_idx = int(np.argmax(probs))
            top_label = self.ecgfounder_labels[top_idx] if self.ecgfounder_labels else f"LABEL_{top_idx}"
            top_prob = float(probs[top_idx])
            all_probs = {
                (self.ecgfounder_labels[i] if i < len(self.ecgfounder_labels) else f"LABEL_{i}"): float(probs[i])
                for i in range(len(probs))
            }
            return {"label": top_label, "prob": top_prob, "all_probs": all_probs}
        except Exception as e:
            logger.warning(f"ECGFounder inference falhou: {e}")
            return None

    # Queenbee superclass labels (5 outputs from the model head)
    QUEENBEE_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

    def _predict_queenbee(self, signal_np):
        signal_ready = self._prepare_signal(signal_np, 5000)
        input_tensor = torch.from_numpy(signal_ready).float().to(self.torch_device)
        with torch.no_grad():
            # ECGTransformer returns (superclass_logits, scp_logits)
            superclass_logits, _ = self.queenbee(input_tensor)
            # Superclass outputs are multi-label → sigmoid per model card
            probs = torch.sigmoid(superclass_logits)[0]
            idx = torch.argmax(probs).item()
            return {"label": self.QUEENBEE_CLASSES[idx], "prob": float(probs[idx])}

    def _extract_signal(self, image_bytes) -> Optional[np.ndarray]:
        if not self.digitizer:
            return None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.torch_device)
            # Restringir a layouts "standard" para evitar que o fallback do LeadIdentifier
            # recaia em cabrera_12x1 (primeira entrada do YAML).
            # Cabrera inverte aVR→-aVR e reordena derivações de membros — erro clínico grave
            # em ECGs padrão (formato dominante no Brasil/EUA).
            got_values = self.digitizer(img_tensor, layout_should_include_substring="standard")
            canonical = got_values.get("signal", {}).get("canonical_lines")
            return canonical.cpu().numpy() if canonical is not None else None
        except Exception as e:
            logger.warning(f"Falha na extração de sinal: {e}")
            return None

    def _extract_rhythm_strip(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Extrai tira de ritmo via digitizador OpenCV (fallback quando Ahus-AIM indisponível)."""
        try:
            from backend.utils.perception.ecg_digitizer_cv import digitize_rhythm_strip
            result = digitize_rhythm_strip(image_bytes)
            if result is not None:
                logger.info(
                    f"OpenCV rhythm strip: {len(result['signal'])} samples, "
                    f"sample_rate≈{result['sample_rate']} Hz"
                )
            return result
        except Exception as e:
            logger.warning(f"Falha no digitizador OpenCV: {e}")
            return None

    def _get_lead_ii(self, signal_np: np.ndarray) -> Optional[np.ndarray]:
        """Extrai Lead II (índice 1) de sinal multi-derivação."""
        try:
            if signal_np.ndim == 3:
                return signal_np[0, 1, :]
            elif signal_np.ndim == 2:
                return signal_np[1, :] if signal_np.shape[0] >= 2 else signal_np[0, :]
            elif signal_np.ndim == 1:
                return signal_np
            return None
        except Exception:
            return None

    def _analyze_with_neurokit2(
        self, signal_1d: np.ndarray, sample_rate: int
    ) -> Optional[Dict[str, Any]]:
        """Análise ECG rule-based via neurokit2.

        Funciona em sinal 1D (tira de ritmo ou Lead II).
        Retorna: hr, rhythm, rr_cv, qrs_duration_ms.
        """
        try:
            import warnings
            import neurokit2 as nk  # lazy import — instalado opcionalmente

            # Suprimir ChainedAssignmentError interno do pandas/neurokit2 (CoW warning)
            # Não afeta resultados — é um detalhe de implementação interna da lib.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*inplace method.*")
                warnings.filterwarnings("ignore", message=".*chained assignment.*")

                signal_clean = nk.ecg_clean(signal_1d, sampling_rate=sample_rate)
                _, rpeaks_dict = nk.ecg_peaks(signal_clean, sampling_rate=sample_rate)
                peaks = rpeaks_dict.get("ECG_R_Peaks", np.array([]))

                if len(peaks) < 2:
                    logger.warning("neurokit2: picos R insuficientes para análise.")
                    return None

                rr_intervals = np.diff(peaks) / sample_rate  # segundos
                mean_rr = float(np.mean(rr_intervals))

                # Frequência cardíaca — apenas se sample_rate for confiável.
                # FC depende de sample_rate; valores < 30 bpm são quase certamente artefato
                # de digitalização óptica (sample_rate estimado, não calibrado por velocidade de papel).
                hr = None
                if mean_rr > 0:
                    hr_raw = int(round(60.0 / mean_rr))
                    if 30 <= hr_raw <= 300:   # < 30 bpm = artefato de estimativa de sample_rate
                        hr = hr_raw

                # Regularidade do ritmo via RMSSD e coeficiente de variação R-R
                rr_ms = rr_intervals * 1000  # milissegundos
                rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0
                rr_cv = float(np.std(rr_intervals) / mean_rr) if mean_rr > 0 else 0.0

                # Classificação do ritmo (thresholds conservadores — complemento visual)
                if rr_cv > 0.20 or rmssd > 80:
                    rhythm = "irregular (avaliar FA ou Flutter)"
                else:
                    rhythm = "regular"

                # Duração do QRS por delineamento (best-effort)
                qrs_duration_ms = None
                try:
                    _, waves = nk.ecg_delineate(
                        signal_clean, rpeaks_dict, sampling_rate=sample_rate, method="cwt"
                    )
                    q_onsets = np.array(waves.get("ECG_Q_Onsets", []), dtype=float)
                    s_offsets = np.array(waves.get("ECG_S_Offsets", []), dtype=float)
                    if len(q_onsets) > 0 and len(q_onsets) == len(s_offsets):
                        valid = [
                            (q, s) for q, s in zip(q_onsets, s_offsets)
                            if not (np.isnan(q) or np.isnan(s))
                        ]
                        if valid:
                            durations = [(s - q) / sample_rate * 1000 for q, s in valid]
                            qrs_duration_ms = int(np.median(durations))
                except Exception as de:
                    logger.debug(f"neurokit2 delineate falhou (não crítico): {de}")

            return {
                "hr": hr,
                "rhythm": rhythm,
                "rmssd": float(rmssd),
                "rr_cv": float(rr_cv),
                "qrs_duration_ms": qrs_duration_ms,
            }

        except ImportError:
            logger.warning("neurokit2 não instalado. Instale com: pip install neurokit2")
            return None
        except Exception as e:
            logger.warning(f"neurokit2 falhou: {e}")
            return None

    def _prepare_signal(self, signal, length):
        if signal.ndim == 2: signal = signal[None, :, :]
        if signal.shape[-1] != length:
            x = np.linspace(0, 1, signal.shape[-1])
            x_new = np.linspace(0, 1, length)
            signal = interp1d(x, signal, axis=-1)(x_new)
        return signal

    def _consensus_vote(self, results: List[Dict[str, Any]]) -> str:
        """E10: Consenso por maioria com desempate por gravidade clínica.
        Modificado para ignorar modelos de baixa confiança ou não fine-tuned.
        """
        if not results:
            return "Indeterminado"

        # Filtrar votos: ignorar modelos com confiança < 60% se houver alternativas,
        # ou se o modelo for explicitamente marcado como baixa confiança (ex: BHF)
        valid_votes = []
        for r in results:
            # Se for o BHF (visual-only) e a probabilidade for baixa, descartar se houver outros
            if r.get("confidence") == "low" and r.get("prob", 0) < 0.75:
                continue
            
            # Se a probabilidade for muito baixa (< 40%), descartar do consenso
            if r.get("prob", 0) < 0.40:
                continue
                
            valid_votes.append(r)

        # Se após o filtro não sobrar nada, usar o melhor chute mas marcar como indeterminado
        if not valid_votes:
            best_r = max(results, key=lambda x: x.get("prob", 0))
            if best_r.get("prob", 0) < 0.50:
                return "Indeterminado"
            return best_r["label"]

        labels = [r["label"] for r in valid_votes]
        unique_labels = set(labels)

        # Contagem de votos
        counts = {label: labels.count(label) for label in unique_labels}
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]

        if len(winners) == 1:
            return winners[0]

        # Desempate: priorizar o achado mais grave clinicamente
        return max(winners, key=lambda l: CLINICAL_SEVERITY.get(l, 0))

    def _estimate_hr(self, signal_np: np.ndarray) -> Optional[int]:
        """E5: Calcula FC real via detecção de picos R na derivação II.

        Assume 5000 amostras = 10 segundos → sample_rate = 500 Hz.
        """
        try:
            # Selecionar derivação II (índice 1 no layout standard 12-lead)
            if signal_np.ndim == 3:
                lead_ii = signal_np[0, 1, :]  # batch, lead, samples
            elif signal_np.ndim == 2:
                lead_ii = signal_np[1, :] if signal_np.shape[0] >= 2 else signal_np[0, :]
            else:
                return None

            sample_rate = 500  # 5000 amostras / 10 segundos
            # Distância mínima entre picos R: ~0.3s (200 bpm max)
            min_distance = int(0.3 * sample_rate)
            # Altura mínima: acima da média + 0.5 * desvio padrão
            height_threshold = np.mean(lead_ii) + 0.5 * np.std(lead_ii)

            peaks, _ = find_peaks(lead_ii, distance=min_distance, height=height_threshold)

            if len(peaks) < 2:
                return None

            # Calcular intervalos R-R em segundos
            rr_intervals = np.diff(peaks) / sample_rate
            mean_rr = np.mean(rr_intervals)

            if mean_rr <= 0:
                return None

            hr = int(round(60.0 / mean_rr))
            # Sanity check: FC entre 20 e 300 bpm
            if 20 <= hr <= 300:
                return hr
            return None
        except Exception as e:
            logger.warning(f"Erro no cálculo de FC: {e}")
            return None

    def get_summary_for_prompt(self, analysis_results: Dict[str, Any], language: str = "en") -> str:
        """E7: Separa mensagem de erro do template de formato.

        Quando há erro ou resultado parcial, instrui o LLM a analisar
        visualmente em vez de usar o formato estruturado HD/FC/Ritmo.

        #8: Accepts language parameter so fallback messages are localised.
        """
        status = analysis_results.get("status", "")

        if "error" in analysis_results and status != "partial":
            # Erro total: instrui análise visual sem formato estruturado
            from prompts.exam_instructions import EXAM_SPECIFIC_PROMPTS
            fb = EXAM_SPECIFIC_PROMPTS.get("ecg_visual_fallback", {})
            fallback = fb.get(language, fb.get("en", ""))
            return f"\n[ECG — ANÁLISE VISUAL NECESSÁRIA]: {fallback}\n"

        if status == "partial_cv":
            # OpenCV rhythm strip + neurokit2 disponível.
            # ATENÇÃO: FC e QRS NÃO são fornecidos aqui — dependem de sample_rate calibrado,
            # que não temos na digitalização óptica. Apenas rr_cv (regularidade do ritmo)
            # é sample_rate-independente e pode ser reportado com segurança.
            nk_rhythm = analysis_results.get("nk_rhythm", "")
            findings = ", ".join(analysis_results.get("findings", []))

            _BHF_MIN_PROB = 0.50
            max_prob = analysis_results.get("max_prob", 0.0)
            raw_rhythm = analysis_results.get("rhythm", "")
            rhythm_display = raw_rhythm
            if max_prob < _BHF_MIN_PROB:
                rhythm_display = (
                    f"Indeterminado (baixa confiança: {max_prob:.0%})"
                    if language == "pt"
                    else f"Indeterminate (low confidence: {max_prob:.0%})"
                )
            nk_rhythm_text = nk_rhythm or ("Não determinado" if language == "pt" else "Not determined")

            # Nota clínica condicional baseada no diagnóstico detectado
            clinical_note = _get_clinical_note(raw_rhythm, language)

            if language == "pt":
                return (
                    "\n[DADOS ECG — OpenCV + neurokit2]:\n"
                    f"DIAGNÓSTICO (modelo visual): {rhythm_display}\n"
                    f"REGULARIDADE DO RITMO (neurokit2): {nk_rhythm_text}\n"
                    f"ACHADOS: {findings}\n"
                    + (f"{clinical_note}\n" if clinical_note else "")
                    + "INSTRUÇÃO: Analise visualmente a imagem e preencha todos os campos "
                    "(FC, Ritmo detalhado, Eixo, Intervalos PR/QRS/QTc, Onda P, Complexo QRS, Onda T). "
                    "Use a REGULARIDADE DO RITMO acima apenas como dado de suporte.\n"
                    "---------------------------\n"
                )
            else:
                return (
                    "\n[ECG DATA — OpenCV + neurokit2]:\n"
                    f"DIAGNOSIS (visual model): {rhythm_display}\n"
                    f"RHYTHM REGULARITY (neurokit2): {nk_rhythm_text}\n"
                    f"FINDINGS: {findings}\n"
                    + (f"{clinical_note}\n" if clinical_note else "")
                    + "INSTRUCTION: Visually analyze the image and fill in all fields "
                    "(HR, detailed Rhythm, Axis, PR/QRS/QTc intervals, P Wave, QRS Complex, T Wave). "
                    "Use RHYTHM REGULARITY above only as supporting data.\n"
                    "---------------------------\n"
                )

        if status == "partial":
            # Apenas BHF visual — pede análise visual com 3 campos
            warnings = "; ".join(analysis_results.get("warnings", []))
            findings = ", ".join(analysis_results.get("findings", []))

            if language == "pt":
                hr_unavail = "Não calculada (sinal indisponível)"
                hr_unknown = "Não determinada"
                tmpl_header = "\n[DADOS PARCIAIS DE TELEMETRIA DO ECG]:\n"
                tmpl_warn = "AVISO"
                tmpl_diag = "DIAGNÓSTICO PARCIAL (apenas modelo visual)"
                tmpl_hr = "FREQUÊNCIA CARDÍACA"
                tmpl_det = "DETALHES"
                tmpl_instr = "INSTRUÇÃO: Preencha APENAS Ritmo / FC estimada / Achado principal. Sem texto adicional."
            else:
                hr_unavail = "Not calculated (signal unavailable)"
                hr_unknown = "Not determined"
                tmpl_header = "\n[PARTIAL ECG TELEMETRY DATA]:\n"
                tmpl_warn = "WARNING"
                tmpl_diag = "PARTIAL DIAGNOSIS (visual model only)"
                tmpl_hr = "HEART RATE"
                tmpl_det = "DETAILS"
                tmpl_instr = "INSTRUCTION: Fill in ONLY Rhythm / Estimated HR / Main finding. No additional text."

            hr_text = f"{analysis_results['hr']} bpm" if analysis_results.get('hr') else hr_unavail

            # #6: Apply probability threshold — BHF with low confidence should not look definitive
            _BHF_MIN_PROB = 0.50
            max_prob = analysis_results.get("max_prob", 0.0)
            raw_rhythm = analysis_results.get('rhythm', '')
            rhythm_display = raw_rhythm
            if max_prob < _BHF_MIN_PROB:
                rhythm_display = (
                    f"Indeterminado (confiança baixa: {max_prob:.0%})"
                    if language == "pt"
                    else f"Indeterminate (low confidence: {max_prob:.0%})"
                )

            # Nota clínica condicional baseada no diagnóstico detectado
            clinical_note = _get_clinical_note(raw_rhythm, language)

            return (
                f"{tmpl_header}"
                f"{tmpl_warn}: {warnings}\n"
                f"{tmpl_diag}: {rhythm_display}\n"
                f"{tmpl_hr}: {hr_text}\n"
                f"{tmpl_det}: {findings}\n"
                + (f"{clinical_note}\n" if clinical_note else "")
                + f"{tmpl_instr}\n"
                "---------------------------\n"
            )

        if status == "error":
            # Todos os modelos falharam — instrui análise visual completa
            from prompts.exam_instructions import EXAM_SPECIFIC_PROMPTS
            fb = EXAM_SPECIFIC_PROMPTS.get("ecg_visual_fallback", {})
            fallback = fb.get(language, fb.get("en", ""))
            return f"\n[ECG — ANÁLISE VISUAL NECESSÁRIA]: {fallback}\n"

        # Success
        if language == "pt":
            hr_unknown = "Não determinada"
            header = "\n[DADOS DE TELEMETRIA DO ECG]:\n"
            lbl_diag = "DIAGNÓSTICO PELOS MODELOS"
            lbl_hr = "FREQUÊNCIA CARDÍACA"
            lbl_det = "DETALHES"
        else:
            hr_unknown = "Not determined"
            header = "\n[ECG TELEMETRY DATA]:\n"
            lbl_diag = "MODEL DIAGNOSIS"
            lbl_hr = "HEART RATE"
            lbl_det = "DETAILS"

        hr_text = f"{analysis_results.get('hr')} bpm" if analysis_results.get('hr') else hr_unknown
        raw_rhythm = analysis_results.get('rhythm', '')
        clinical_note = _get_clinical_note(raw_rhythm, language)
        return (
            f"{header}"
            f"{lbl_diag}: {raw_rhythm}\n"
            f"{lbl_hr}: {hr_text}\n"
            f"{lbl_det}: {', '.join(analysis_results.get('findings', []))}\n"
            + (f"{clinical_note}\n" if clinical_note else "")
            + "---------------------------\n"
        )

_ecg_analyzer = None
_ecg_analyzer_error = None

def get_ecg_analyzer() -> ECGAnalyzer:
    """Lazy factory — instantiates ECGAnalyzer on first call."""
    global _ecg_analyzer, _ecg_analyzer_error
    if _ecg_analyzer is None and _ecg_analyzer_error is None:
        try:
            _ecg_analyzer = ECGAnalyzer()
        except Exception as e:
            _ecg_analyzer_error = str(e)
            logger.exception(f"Failed to initialize ECGAnalyzer: {e}")
    if _ecg_analyzer_error is not None and _ecg_analyzer is None:
        raise RuntimeError(f"ECGAnalyzer init failed: {_ecg_analyzer_error}")
    return _ecg_analyzer

def is_ecg_analyzer_loaded() -> bool:
    return _ecg_analyzer is not None

def ecg_analyzer_error() -> Optional[str]:
    return _ecg_analyzer_error
