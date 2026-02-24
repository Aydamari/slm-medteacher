#!/usr/bin/env bash
# One-shot GitHub push for SLM MedTeacher
set -euo pipefail

REPO_DIR="/home/portugalfaria/Insync/afaria@id.uff.br/Google Drive/MED/AI/Github/slm-medteacher"
GITHUB_USER="Aydamari"
REPO_NAME="slm-medteacher"

echo "→ Creating GitHub repository..."
gh repo create "$GITHUB_USER/$REPO_NAME" \
    --public \
    --description "Local AI Medical Teaching Assistant — MedGemma 1.5 4B + ECG/X-ray perception pipeline" \
    2>/dev/null || echo "  (repo may already exist — continuing)"

echo "→ Entering project directory..."
cd "$REPO_DIR"

echo "→ Adding weight-file exclusions to .gitignore..."
cat >> .gitignore << 'GITIGNORE'

# Large model weights — exceed GitHub 100 MB limit
# Download instructions: see UPLOAD_GUIDE.md
*.pth
*.h5
*.safetensors
backend/models/ecg_segmentation_unet.pt
backend/utils/perception/ecg_image_kit/roi/yolov7/yolov7_custom.pt
backend/models/weights/queenbee/ecg_transformer_best.pt
GITIGNORE

echo "→ Renaming branch to main..."
git branch -M main 2>/dev/null || true

echo "→ Staging all files..."
git add .

echo "→ Committing..."
git commit -m "SLM MedTeacher — HAI-DEF Models Competition

Local AI medical teaching assistant using google/medgemma-1.5-4b-it (MedGemma 1.5).
ECG: nnU-Net SOTA digitizer + CODE-15 + ECGFounder + Queenbee + NeuroKit2 ensemble.
X-ray: TorchXRayVision x4 + MedGemma 1.5 native vision (HAI-DEF native).
Privacy-first, offline-capable, open source.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo "→ Setting remote origin..."
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git" 2>/dev/null \
    || git remote set-url origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "→ Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Repository live at: https://github.com/$GITHUB_USER/$REPO_NAME"
