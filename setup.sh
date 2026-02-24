#!/usr/bin/env bash
# =============================================================================
# SLM MedTeacher — One-Line Installer
# Works on Linux and macOS. Requires Python 3.10+ and Ollama.
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Author: Aydamari Faria Jr.
# =============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
info() { echo -e "${BLUE}→${NC} $*"; }
header() { echo -e "\n${BOLD}$*${NC}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      SLM MedTeacher — Installer          ║${NC}"
echo -e "${BOLD}║  Local AI Medical Teaching Assistant     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
header "Step 1/7 — Checking Python"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=${version%%.*}
        minor=${version##*.}
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            ok "Found $cmd (Python $version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "Python 3.10 or newer is required. Install from https://python.org"
fi

# ── 2. Ollama check ───────────────────────────────────────────────────────────
header "Step 2/7 — Checking Ollama"

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
    ok "Ollama found: $OLLAMA_VER"
else
    warn "Ollama not found."
    echo "   Install from: https://ollama.ai/download"
    echo "   Linux one-liner: curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    read -r -p "   Continue without Ollama? (requires manual install later) [y/N]: " choice
    [[ "$choice" =~ ^[Yy]$ ]] || fail "Please install Ollama first, then re-run setup.sh"
fi

# ── 3. Virtual environment ────────────────────────────────────────────────────
header "Step 3/7 — Creating virtual environment"

if [ -d "venv" ]; then
    warn "Virtual environment already exists — skipping creation."
    ok "Using existing venv/"
else
    "$PYTHON_CMD" -m venv venv
    ok "Virtual environment created at venv/"
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate
ok "Virtual environment activated"

# ── 4. Dependencies ───────────────────────────────────────────────────────────
header "Step 4/7 — Installing Python dependencies"

if [ ! -f requirements.txt ]; then
    fail "requirements.txt not found. Make sure you are in the SLM MedTeacher root directory."
fi

pip install --upgrade pip --quiet
pip install -r requirements.txt

ok "Python dependencies installed"

# ── 5. Ollama model ───────────────────────────────────────────────────────────
header "Step 5/7 — Pulling MedGemma 1.5 4B model"

MODEL_4B="thiagomoraes/medgemma-1.5-4b-it:Q4_K_M"

if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | grep -q "medgemma-1.5-4b-it"; then
        ok "MedGemma 1.5 4B already downloaded."
    else
        info "Downloading $MODEL_4B (~3 GB)..."
        info "This may take 5–20 minutes depending on your connection."
        ollama pull "$MODEL_4B" && ok "MedGemma 1.5 4B downloaded." || warn "Model download failed — you can run 'ollama pull $MODEL_4B' manually."
    fi
else
    warn "Ollama not available — skipping model download."
    echo "   Run manually after installing Ollama:"
    echo "   ollama pull $MODEL_4B"
fi

# ── 6. Model weights (perception pipeline) ────────────────────────────────────
header "Step 6/7 — Checking ML model weights"

echo ""
echo "   The perception pipeline requires several pre-trained model weight files."
echo "   These cannot be auto-downloaded due to access controls."
echo ""

WEIGHTS_DIR="backend/models/weights"
CODE15_PATH="backend/models/resnet_code15.h5"

check_weight() {
    local path="$1"
    local name="$2"
    local size_req="${3:-1000000}"  # 1MB minimum
    if [ -f "$path" ]; then
        size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)
        if [ "$size" -gt "$size_req" ]; then
            ok "$name present ($((size / 1024 / 1024)) MB)"
        else
            warn "$name found but appears to be an LFS pointer — re-download required."
        fi
    else
        warn "$name NOT FOUND at $path"
    fi
}

check_weight "$CODE15_PATH"                                       "CODE-15 (ResNet)"      5000000
check_weight "$WEIGHTS_DIR/ecgfounder/12_lead_ECGFounder.pth"    "ECGFounder"            50000000
check_weight "$WEIGHTS_DIR/queenbee/ecg_transformer_best.pt"      "Queenbee"              10000000
check_weight "backend/models/ecg_segmentation_unet.pt"            "Ahus-AIM Segmentation" 80000000
check_weight "$WEIGHTS_DIR/nnunet_ecg_sota/repo/models/M3/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth" \
             "nnU-Net M3 (SOTA ECG)"  400000000

echo ""
echo "   ─── Weight download instructions ─────────────────────────────────────"
echo ""
echo "   ECGFounder + Queenbee (HuggingFace Hub):"
echo "     pip install huggingface_hub"
echo "     python -c \""
echo "       from huggingface_hub import snapshot_download"
echo "       snapshot_download('thibaultneveu/queenbee', local_dir='$WEIGHTS_DIR/queenbee')"
echo "     \""
echo ""
echo "   nnU-Net M3 checkpoint:"
echo "     Download from the ECG-Digitiser PhysioNet 2024 GitHub releases"
echo "     (see backend/models/weights/nnunet_ecg_sota/repo/README.md)"
echo ""
echo "   CODE-15:"
echo "     Download resnet_code15.h5 from the CODE-15 project GitHub"
echo "     Place at: $CODE15_PATH"
echo ""

# ── 7. Create required directories ───────────────────────────────────────────
header "Step 7/7 — Creating directories"

mkdir -p sessions/exports
mkdir -p backend/models/weights/ecgfounder
mkdir -p backend/models/weights/queenbee
mkdir -p backend/models/weights/nnunet_ecg_sota/repo/models/M3
ok "Directories ready"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║           Setup Complete!                ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "   To start the server:"
echo ""
echo -e "     ${GREEN}source venv/bin/activate${NC}"
echo -e "     ${GREEN}uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload${NC}"
echo ""
echo "   Then open: http://localhost:8000/medteacher.html"
echo ""
echo "   For cloud LLMs (Gemini 2.5, etc.), set your OpenRouter API key:"
echo -e "     ${GREEN}python installers/setup_secrets.py${NC}"
echo ""
echo "   Health check:"
echo -e "     ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
