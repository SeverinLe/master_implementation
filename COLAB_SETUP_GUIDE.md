# Google Colab Setup Guide for Cursor

## Step 1: Sign In to Colab Extension

1. **Open Command Palette** in Cursor:
   - Mac: `Cmd + Shift + P`
   - Windows/Linux: `Ctrl + Shift + P`

2. Type: `Colab: Sign In` and press Enter

3. A browser window will open - **Sign in with your Google account**

4. Grant permissions when prompted

5. You'll see "✓ Signed in" in Cursor's status bar

---

## Step 2: Upload Your Project to Colab

### Option A: Use the Notebook I Created

1. Open `colab_training.ipynb` in Cursor

2. **Right-click** the notebook → Select **"Open in Colab"**
   - Or use Command Palette: `Colab: Open in Colab`

3. The notebook opens in your browser at `colab.research.google.com`

### Option B: Manual Upload

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Select `colab_training.ipynb` from your project

---

## Step 3: Prepare Your Data

Before running the notebook, you need to upload `train_reps.pt`:

### Quick Method (Direct Upload):
- Run the upload cell in the notebook
- Select your `data/tiny-imagenet-200/train_reps.pt` file (826MB)
- Wait 5-10 minutes for upload

### Better Method (Use Google Drive):
1. Upload `train_reps.pt` to Google Drive beforehand
2. In Colab, mount Drive and copy the file
3. Much faster for repeated sessions!

```bash
# Upload to Drive once:
# 1. Go to drive.google.com
# 2. Create folder: rcdm_data
# 3. Upload train_reps.pt there

# Then in Colab:
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/rcdm_data/train_reps.pt /content/master_implementation/data/tiny-imagenet-200/
```

---

## Step 4: Package Your Code

Create a ZIP of your project (exclude large files):

```bash
cd /Users/severin/master_implementation
zip -r master_implementation.zip \
    rcdm/ \
    scripts/ \
    guided_diffusion/ \
    test_encoder.py \
    sanitycheckSL.py \
    -x "*.pt" "*.pth" "*/__pycache__/*" ".venv/*" "data/*"
```

Upload this ZIP in the Colab notebook (Cell 3).

---

## Step 5: Run Training

1. **Select GPU Runtime**:
   - In Colab: `Runtime → Change runtime type → GPU (T4)`
   - Click Save

2. **Run cells in order** (click the play button on each cell):
   - Cell 1: Check GPU ✓
   - Cell 2: Mount Drive ✓
   - Cell 3: Upload code ZIP ✓
   - Cell 4: Install dependencies ✓
   - Cell 5: Upload data ✓
   - Cell 6: Verify setup ✓
   - Cell 7: **Start training!** 🚀

3. **Monitor progress**:
   - Training logs appear in cell output
   - Run Cell 8 to check GPU usage
   - Checkpoints save to Google Drive

---

## Benefits of Using Colab

| Feature | Local (CPU) | Colab (GPU) |
|---------|-------------|-------------|
| **Speed** | ~100 steps/hour | ~1000-2000 steps/hour |
| **Batch Size** | 16 | 64-128 |
| **Cost** | Free | Free (with limits) |
| **Training Time** | Weeks | Days |

---

## Colab Limitations & Workarounds

### 1. Session Timeout (12 hours max)
**Solution**: Save checkpoints frequently (`--save_interval 5000`)

```bash
# Resume from checkpoint:
!python scripts/train.py \
    --resume /content/drive/MyDrive/rcdm_checkpoints/model_25000.pt \
    --device cuda
```

### 2. Disconnections
**Solution**: Keep browser tab open, use Google Drive for checkpoints

### 3. GPU Availability
**Solution**: Colab Pro ($10/month) for priority access

---

## Troubleshooting

### "No GPU detected"
- Go to `Runtime → Change runtime type → GPU`
- Restart runtime: `Runtime → Restart runtime`

### "ModuleNotFoundError: No module named 'rcdm'"
```python
import sys
sys.path.insert(0, '/content/master_implementation')
```

### "Out of Memory"
- Reduce batch size: `--batch_size 32`
- Restart runtime to clear memory

### Training Too Slow
- Check GPU usage: `!nvidia-smi`
- Verify you're using GPU: `--device cuda`
- Increase batch size if GPU not fully utilized

---

## Next Steps After Training

1. **Download checkpoints** from Google Drive
2. **Generate samples** using your trained model
3. **Evaluate** on validation set

Let me know if you need help creating sampling/evaluation scripts!
