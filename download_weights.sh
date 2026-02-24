#!/usr/bin/env bash
# =============================================================================
# SLM MedTeacher — Model Weights Downloader
#
# Downloads all ML model weights required for the perception pipeline.
# Run this AFTER setup.sh (requires the venv and huggingface_hub).
#
# Usage:
#   source venv/bin/activate
#   chmod +x download_weights.sh && ./download_weights.sh
#
# What this downloads (~1.5 GB total):
#   • CODE-15 ResNet       25 MB  — GitHub Release
#   • Ahus-AIM U-Net       87 MB  — GitHub Release
#   • nnU-Net M3 checkpoint 453 MB — GitHub Release
#   • YOLOv7 ROI detector  71 MB  — GitHub Release
#   • Queenbee Transformer  ~200 MB — HuggingFace (Trustcat/queenbee-ecg-transformer)
#   • ECGFounder PKU        ~600 MB — HuggingFace (PKUDigitalHealth/ECGFounder)
#
# TorchXRayVision (4 X-ray models) and PaddleOCR auto-download on first use.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
info() { echo -e "${BLUE}→${NC} $*"; }

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   SLM MedTeacher — Download Weights     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
    fail "wget or curl is required. Install with: sudo apt install wget"
fi

DOWNLOADER="wget"
if ! command -v wget &>/dev/null; then
    DOWNLOADER="curl"
    warn "wget not found — using curl"
fi

download() {
    local url="$1"
    local dest="$2"
    local name="$3"
    if [ -f "$dest" ]; then
        local size
        size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
        if [ "$size" -gt 1000000 ]; then
            ok "$name already present — skipping."
            return
        fi
    fi
    info "Downloading $name..."
    mkdir -p "$(dirname "$dest")"
    if [ "$DOWNLOADER" = "wget" ]; then
        wget -q --show-progress -c "$url" -O "$dest" || fail "Download failed: $name"
    else
        curl -L --progress-bar -C - "$url" -o "$dest" || fail "Download failed: $name"
    fi
    ok "$name downloaded."
}

RELEASE="https://github.com/Aydamari/slm-medteacher/releases/download/v1.0.0"

# ── 1. GitHub Release assets ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[1/3] GitHub Release weights${NC}"

download "$RELEASE/resnet_code15.h5" \
         "backend/models/resnet_code15.h5" \
         "CODE-15 ResNet (resnet_code15.h5)"

download "$RELEASE/ecg_segmentation_unet.pt" \
         "backend/models/ecg_segmentation_unet.pt" \
         "Ahus-AIM Segmentation U-Net"

download "$RELEASE/yolov7_custom.pt" \
         "backend/utils/perception/ecg_image_kit/roi/yolov7/yolov7_custom.pt" \
         "YOLOv7 ROI detector"

# ── 2. nnU-Net SOTA (repo structure + checkpoint) ─────────────────────────────
echo ""
echo -e "${BOLD}[2/3] nnU-Net SOTA ECG digitizer${NC}"

NNUNET_REPO="backend/models/weights/nnunet_ecg_sota/repo"
CHECKPOINT_DIR="$NNUNET_REPO/models/M3/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/fold_all"

if [ ! -d "$NNUNET_REPO/.git" ]; then
    info "Cloning ECG-Digitiser repo (directory structure, no large files)..."
    git clone --depth 1 --filter=blob:limit=5m \
        https://github.com/felixkrones/ECG-Digitiser "$NNUNET_REPO" \
        || fail "Failed to clone ECG-Digitiser. Check internet connection."
    ok "ECG-Digitiser repo cloned."
else
    ok "ECG-Digitiser repo already present."
fi

download "$RELEASE/checkpoint_final.pth" \
         "$CHECKPOINT_DIR/checkpoint_final.pth" \
         "nnU-Net M3 checkpoint (~453 MB)"

# ── 3. HuggingFace models ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/3] HuggingFace models (Queenbee + ECGFounder)${NC}"
echo "      This downloads ~800 MB and caches to ~/.cache/huggingface/"
echo ""

if ! python3 -c "import huggingface_hub" &>/dev/null; then
    warn "huggingface_hub not found — installing..."
    pip install -q huggingface_hub
fi

python3 - <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

models = [
    ("Trustcat/queenbee-ecg-transformer", "Queenbee ECG Transformer"),
    ("PKUDigitalHealth/ECGFounder",        "ECGFounder (PKU)"),
]

for repo_id, name in models:
    print(f"\n  → {name} ({repo_id})")
    try:
        path = snapshot_download(repo_id)
        print(f"  ✓ {name} cached at: {path}")
    except Exception as e:
        print(f"  ✗ {name} failed: {e}", file=sys.stderr)
        sys.exit(1)
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         Weights Downloaded!              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "   Next step:"
echo -e "     ${GREEN}uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload${NC}"
echo ""
echo "   Then open: http://localhost:8000/medteacher.html"
echo ""
