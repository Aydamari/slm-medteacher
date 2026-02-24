"""
ECG Rhythm Strip Digitizer — OpenCV Rule-Based Fallback

Extracts a 1D waveform from the rhythm strip of an ECG image when the
primary Ahus-AIM digitizer (requires LFS weights) is unavailable.

Pipeline:
  1. Decode image → grayscale
  2. Crop rhythm strip region (bottom ~30% of image)
  3. CLAHE contrast enhancement
  4. Invert if background is light (standard ECG paper)
  5. Adaptive threshold to isolate ECG trace
  6. Column-wise centre-of-mass waveform extraction
  7. Normalise to zero-mean, unit-variance
  8. Estimate sample_rate = strip_width // 10  (assumes 10-second strip)

Returns:
  {"signal": np.ndarray (1D float), "sample_rate": int, "method": "opencv_rule_based"}
  or None if extraction fails.
"""

import logging
from typing import Optional, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fraction of image height used as rhythm strip region (bottom portion)
_STRIP_FRACTION = 0.30
# Minimum strip width to attempt digitization
_MIN_STRIP_WIDTH = 100
# Minimum sample rate floor (Hz)
_MIN_SAMPLE_RATE = 100


def digitize_rhythm_strip(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Extract ECG waveform from rhythm strip using OpenCV.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        dict with "signal" (1D np.ndarray), "sample_rate" (int), "method" (str)
        or None if extraction fails.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("OpenCV digitizer: could not decode image bytes.")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if w < _MIN_STRIP_WIDTH:
            logger.warning(f"OpenCV digitizer: image too narrow ({w}px).")
            return None

        # 1. Crop rhythm strip (bottom 30%)
        strip_start = int(h * (1.0 - _STRIP_FRACTION))
        strip = gray[strip_start:, :]
        strip_h, strip_w = strip.shape

        # 2. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(strip)

        # 3. Invert if background is light (standard ECG paper is white)
        mean_val = float(np.mean(enhanced))
        if mean_val > 128:
            enhanced = cv2.bitwise_not(enhanced)

        # 4. Adaptive threshold — isolate ECG trace (dark pixels after invert)
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=-5,
        )

        # 5. Column-wise centre-of-mass waveform extraction
        signal = _extract_waveform_com(binary, strip_h, strip_w)
        if signal is None:
            return None

        # 6. Normalise
        std = float(np.std(signal))
        if std > 0:
            signal = (signal - float(np.mean(signal))) / std
        else:
            signal = signal - float(np.mean(signal))

        # 7. Estimate sample rate: standard 12-lead ECG = 25 mm/s, 10 s strip
        sample_rate = max(_MIN_SAMPLE_RATE, strip_w // 10)

        logger.info(
            f"OpenCV digitizer: extracted {len(signal)} samples, "
            f"sample_rate≈{sample_rate} Hz, strip={strip_w}×{strip_h}px"
        )
        return {
            "signal": signal.astype(np.float32),
            "sample_rate": sample_rate,
            "method": "opencv_rule_based",
        }

    except Exception as exc:
        logger.warning(f"OpenCV digitizer failed: {exc}")
        return None


def _extract_waveform_com(
    binary: np.ndarray, strip_h: int, strip_w: int
) -> Optional[np.ndarray]:
    """Column-wise centre-of-mass extraction from a binarised ECG strip."""
    signal = np.zeros(strip_w, dtype=np.float64)
    indices = np.arange(strip_h, dtype=np.float64)
    active_cols = 0

    for col in range(strip_w):
        column = binary[:, col].astype(np.float64)
        total = column.sum()
        if total > 0:
            com = float(np.dot(indices, column) / total)
            # Invert: lower centre-of-mass (bottom of strip) → lower voltage
            signal[col] = strip_h - com
            active_cols += 1
        else:
            signal[col] = strip_h / 2.0  # baseline fallback

    # Require at least 50% of columns to have active trace
    if active_cols < strip_w * 0.50:
        logger.warning(
            f"OpenCV digitizer: only {active_cols}/{strip_w} active columns "
            "(trace too faint or image too noisy)."
        )
        return None

    return signal
