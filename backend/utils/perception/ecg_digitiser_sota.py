import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import rotate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "backend" / "models" / "weights" / "nnunet_ecg_sota"
MODEL_FOLDER_M3 = WEIGHTS_DIR / "repo" / "models" / "M3"

# ---------------------------------------------------------------------------
# Constants from ECG-Digitiser/config.py (Felix Krones – PhysioNet 2024)
# ---------------------------------------------------------------------------
DATASET_NAME = "Dataset500_Signals"
FREQUENCY = 500
LONG_SIGNAL_LENGTH_SEC = 10
SHORT_SIGNAL_LENGTH_SEC = 2.5

LEAD_LABEL_MAPPING = {
    "I": 1, "II": 2, "III": 3,
    "aVR": 4, "aVL": 5, "aVF": 6,
    "V1": 7, "V2": 8, "V3": 9,
    "V4": 10, "V5": 11, "V6": 12,
}

# Canonical 12-lead order expected by downstream classifiers
CANONICAL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"]

Y_SHIFT_RATIO = {
    "I":   12.6  / 21.59,
    "II":   9.0  / 21.59,
    "III":  5.4  / 21.59,
    "aVR": 12.6  / 21.59,
    "aVL":  9.0  / 21.59,
    "aVF":  5.4  / 21.59,
    "V1":  12.59 / 21.59,
    "V2":   9.0  / 21.59,
    "V3":   5.4  / 21.59,
    "V4":  12.59 / 21.59,
    "V5":   9.0  / 21.59,
    "V6":   5.4  / 21.59,
    "full": 2.1  / 21.59,
}


# ---------------------------------------------------------------------------
# Weight readiness check
# ---------------------------------------------------------------------------

def _nnunet_weights_ready() -> bool:
    """Return True only if M3 fold_all checkpoint_final.pth is a real file (> 1 MB)."""
    checkpoint = (
        MODEL_FOLDER_M3
        / "nnUNet_results"
        / DATASET_NAME
        / "nnUNetTrainer__nnUNetPlans__2d"
        / "fold_all"
        / "checkpoint_final.pth"
    )
    return checkpoint.exists() and checkpoint.stat().st_size > 1_000_000


# ---------------------------------------------------------------------------
# Hough Transform – rotation detection
# ---------------------------------------------------------------------------

def _get_lines(np_image: np.ndarray, threshold=1200, rho_resolution=1):
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return cv2.HoughLines(edges, rho_resolution, np.pi / 180, threshold, None, 0, 0)


def _filter_lines(lines, degree_window=30, parallelism_count=3, parallelism_window=2):
    if lines is None:
        return None
    par_rad = np.deg2rad(parallelism_window)
    filtered = [
        (rho, theta)
        for line in lines
        for rho, theta in line
        if abs(90 - theta * 180 / np.pi) < degree_window
    ]
    parallel = [
        (rho, theta)
        for rho, theta in filtered
        if sum(
            1 for _, ct in filtered
            if abs(theta - ct) < par_rad or abs((theta - ct) - np.pi) < par_rad
        ) >= parallelism_count
    ]
    return np.array(parallel)[:, np.newaxis, :] if parallel else None


def _get_rotation_angle(np_image: np.ndarray) -> float:
    lines    = _get_lines(np_image, threshold=1200)
    filtered = _filter_lines(lines, degree_window=30, parallelism_count=3, parallelism_window=2)
    if filtered is None:
        return 0.0
    angles = [-(90 - line[1] * 180 / np.pi) for line in filtered[:, 0, :]]
    return round(float(np.median(angles)), 4)


# ---------------------------------------------------------------------------
# nnU-Net segmentation via CLI subprocess
# ---------------------------------------------------------------------------

def _predict_mask_nnunet(image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Write image to a temp dir, call nnUNetv2_predict, return the mask tensor.
    Uses M3/fold_all (best PhysioNet-2024 model).
    """
    temp_in  = Path(tempfile.mkdtemp())
    temp_out = Path(tempfile.mkdtemp())
    try:
        img_path  = temp_in  / "00000_temp_0000.png"
        mask_path = temp_out / "00000_temp.png"

        write_png(image_tensor.cpu(), str(img_path))

        env = os.environ.copy()
        env["nnUNet_results"] = str(MODEL_FOLDER_M3 / "nnUNet_results")

        nnunet_bin = Path(sys.executable).parent / "nnUNetv2_predict"
        base_cmd = [
            str(nnunet_bin),
            "-d", DATASET_NAME,
            "-i", str(temp_in),
            "-o", str(temp_out),
            "-f", "all",
            "-tr", "nnUNetTrainer",
            "-c", "2d",
            "-p", "nnUNetPlans",
            # Forçar 1 worker de pré-processamento para evitar crash de RAM em máquinas
            # com muitos cores (nnUNet detecta 24 cores e tenta spawnar processos demais).
            "--num_processes_preprocessing", "1",
            "--num_processes_segmentation_export", "1",
        ]
        if not torch.cuda.is_available():
            base_cmd += ["-device", "cpu"]

        proc = subprocess.run(
            base_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            logger.warning(f"nnUNetv2_predict falhou (rc={proc.returncode}): {proc.stderr[-400:]}")
            return None

        if not mask_path.exists():
            logger.warning("nnUNetv2_predict não gerou máscara de saída.")
            return None

        return read_image(str(mask_path))

    except subprocess.TimeoutExpired:
        logger.warning("nnUNetv2_predict timeout (300s).")
        return None
    except Exception as e:
        logger.error(f"Erro na predição nnU-Net: {e}")
        return None
    finally:
        shutil.rmtree(temp_in,  ignore_errors=True)
        shutil.rmtree(temp_out, ignore_errors=True)


# ---------------------------------------------------------------------------
# Signal vectorization helpers
# ---------------------------------------------------------------------------

def _cut_to_mask(img: torch.Tensor, mask: torch.Tensor, return_y1: bool = False):
    coords = torch.where(mask[0] >= 1)
    if len(coords[0]) == 0:
        return (img, 0, 0) if return_y1 else img
    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()
    img_cut = img[:, y_min:y_max + 1, x_min:x_max + 1]
    if return_y1:
        return img_cut, y_min, x_min
    return img_cut


def _cut_binary(mask_to_use: torch.Tensor, image_rotated: torch.Tensor):
    signal_masks     = {}
    signal_positions = {}
    for lead_name, lead_value in LEAD_LABEL_MAPPING.items():
        binary_mask = torch.where(mask_to_use == lead_value, 1, 0)
        if binary_mask.sum() > 0:
            _, y1, x1    = _cut_to_mask(image_rotated, binary_mask, True)
            signal_mask  = _cut_to_mask(binary_mask,   binary_mask)
            signal_masks[lead_name]     = signal_mask
            signal_positions[lead_name] = {"y1": y1, "x1": x1}
        else:
            signal_masks[lead_name]     = None
            signal_positions[lead_name] = None
    return signal_masks, signal_positions


def _vectorise(image_rotated, mask, y1_scalar, sec_per_pixel, mV_per_pixel, lead):
    total_sec = round(float(sec_per_pixel * mask.shape[2]), 1)
    if total_sec > (LONG_SIGNAL_LENGTH_SEC / 2):
        total_seconds  = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = Y_SHIFT_RATIO["full"]
    else:
        total_seconds  = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = Y_SHIFT_RATIO[lead]

    values_needed = int(total_seconds * FREQUENCY)

    non_zero_mean = torch.tensor([
        (torch.nonzero(mask[0, :, i]).float().mean().item()
         if mask[0, :, i].any() else 0.0)
        for i in range(mask.shape[2])
    ])

    signal_cropped_shifted = (1 - y_shift_ratio_) * image_rotated.shape[1] - y1_scalar
    predicted_signal = (signal_cropped_shifted - non_zero_mean) * mV_per_pixel

    n = predicted_signal.shape[0]
    resampled = F.interpolate(
        predicted_signal.view(1, 1, n).float(),
        size=values_needed,
        mode="linear",
        align_corners=False,
    ).view(-1)
    return resampled.numpy()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ECGDigitiserSOTA:
    """
    SOTA ECG digitizer (PhysioNet Challenge 2024 winner approach).
    Segmentation: nnU-Net M3/fold_all (Dataset500_Signals).
    Signal extraction: geometric vectorization (Hough + mask centroid).
    """

    def __init__(self, device: str = "cuda"):
        self.device   = device if torch.cuda.is_available() else "cpu"
        self.is_ready = _nnunet_weights_ready()
        if self.is_ready:
            logger.info("Motor SOTA (nnU-Net): pesos M3/fold_all materializados e prontos.")
        else:
            logger.warning("Motor SOTA (nnU-Net): pesos não materializados — motor indisponível.")

    def digitize(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Digitize an ECG image.

        Returns:
            np.ndarray of shape (12, N) in mV at 500 Hz, lead order per
            CANONICAL_LEAD_ORDER, or None on failure.
        """
        if not self.is_ready:
            return None

        try:
            # --- 1. Decode ---
            nparr  = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                logger.warning("SOTA: falha ao decodificar bytes da imagem.")
                return None
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Torch tensor (C, H, W) uint8
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)[:3]

            # --- 2. Rotation correction ---
            rot_angle = _get_rotation_angle(img_rgb)
            if not np.isnan(rot_angle) and abs(rot_angle) > 0.1:
                img_tensor = rotate(img_tensor, rot_angle)
            image_rotated = img_tensor

            # --- 3. Segmentation ---
            mask = _predict_mask_nnunet(image_rotated)
            if mask is None:
                return None

            # --- 4. Per-lead binary masks ---
            signal_masks, signal_positions = _cut_binary(mask, image_rotated)

            # --- 5. Spatial calibration ---
            x_pixel_list = [v.shape[2] for v in signal_masks.values() if v is not None]
            if not x_pixel_list:
                logger.warning("SOTA: nenhuma derivação detectada na máscara de segmentação.")
                return None

            x_median          = float(np.median(x_pixel_list))
            x_below_2x        = [v for v in x_pixel_list if v < 2 * x_median]
            sec_per_pixel     = 2.5 / float(np.mean(x_below_2x))
            mm_per_pixel      = 25.0 * sec_per_pixel
            mV_per_pixel      = mm_per_pixel / 10.0

            # --- 6. Vectorize ---
            num_samples = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)  # 5000
            signals_predicted = {}
            for lead, mask_lead in signal_masks.items():
                if mask_lead is not None and signal_positions[lead] is not None:
                    y1  = signal_positions[lead]["y1"]
                    sig = _vectorise(image_rotated, mask_lead, y1,
                                     sec_per_pixel, mV_per_pixel, lead)
                    if len(sig) < num_samples:
                        padded = np.full(num_samples, np.nan, dtype=np.float32)
                        padded[:len(sig)] = sig
                        signals_predicted[lead] = padded
                    else:
                        signals_predicted[lead] = sig[:num_samples].astype(np.float32)
                else:
                    signals_predicted[lead] = np.zeros(num_samples, dtype=np.float32)

            # --- 7. Range check / normalization ---
            all_vals   = np.concatenate(list(signals_predicted.values()))
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size > 0 and (np.max(np.abs(finite_vals)) > 10):
                logger.info("SOTA: sinal fora do range mV — normalizando para [-1, 1].")
                max_v = np.nanmax(all_vals)
                min_v = np.nanmin(all_vals)
                rng   = max_v - min_v if (max_v - min_v) > 0 else 1.0
                for lead in signals_predicted:
                    signals_predicted[lead] = (signals_predicted[lead] - min_v) / rng * 2 - 1

            # --- 8. Assemble (12, 5000) in canonical order ---
            result = np.zeros((12, num_samples), dtype=np.float32)
            for i, lead in enumerate(CANONICAL_LEAD_ORDER):
                if lead in signals_predicted:
                    result[i] = np.nan_to_num(signals_predicted[lead], nan=0.0)

            detected_leads = [l for l, v in signal_masks.items() if v is not None]
            logger.info(f"SOTA: sinal extraído shape={result.shape}, derivações={detected_leads}")
            return result

        except Exception as e:
            logger.error(f"Falha no motor SOTA: {e}", exc_info=True)
            return None


def get_sota_digitiser(device: str = "cuda") -> ECGDigitiserSOTA:
    return ECGDigitiserSOTA(device=device)
