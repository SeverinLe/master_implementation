#!/bin/bash
# prepare_for_colab.sh
# Creates a ZIP package of your project for uploading to Google Colab

echo "📦 Preparing project for Google Colab..."

# Navigate to project root
cd "$(dirname "$0")"

# Create ZIP excluding large files and unnecessary folders
echo "Creating master_implementation.zip..."

zip -r master_implementation.zip \
    rcdm/ \
    scripts/ \
    guided_diffusion/ \
    test_encoder.py \
    sanitycheckSL.py \
    colab_training.ipynb \
    COLAB_SETUP_GUIDE.md \
    -x "*.pt" "*.pth" "*.pkl" "*.npz" \
       "*/__pycache__/*" \
       ".venv/*" \
       "data/*" \
       "checkpoints/*" \
       ".git/*" \
       ".DS_Store" \
       "*.pyc"

# Check result
if [ -f "master_implementation.zip" ]; then
    size=$(du -h master_implementation.zip | cut -f1)
    echo "✓ Created master_implementation.zip ($size)"
    echo ""
    echo "Next steps:"
    echo "1. Open colab_training.ipynb in Cursor"
    echo "2. Right-click → 'Open in Colab' (or Cmd+Shift+P → 'Colab: Open in Colab')"
    echo "3. Upload this ZIP file when prompted in the notebook"
    echo ""
    echo "Don't forget to upload train_reps.pt separately (it's 826MB)!"
else
    echo "❌ Failed to create ZIP file"
    exit 1
fi
