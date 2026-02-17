# 🔍 Image Classifier

**Production-grade multi-class image classification using transfer learning.**  
Built with [fast.ai](https://docs.fast.ai) + PyTorch. ResNet-50 backbone. One command to train, one command to predict.

![CI](https://github.com/YOUR_USERNAME/image-classifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Architecture** | ResNet-50 (or 34 / EfficientNet-B0 / B3) |
| **Transfer learning** | ImageNet pretrained — works with as few as 50 images/class |
| **Two-phase training** | Freeze → head train → unfreeze → discriminative LR fine-tune |
| **LR finder** | Smith (2017) LR range test — no manual LR tuning |
| **Augmentation** | Flip, rotate, zoom, lighting, perspective warp |
| **Regularisation** | MixUp, label smoothing, dropout, AdamW weight decay |
| **Mixed precision** | FP16 on CUDA — 2× faster, 2× lower VRAM |
| **Export** | Self-contained `.pkl` — single file inference, zero config |
| **REST API** | FastAPI server, Docker-ready |
| **Reproducible** | Fixed random seed across Python / NumPy / PyTorch |

---

## 🗂 Project Structure

```
image-classifier/
│
├── config/
│   └── config.yaml              ← all hyperparameters
│
├── data/
│   └── dataset/                 ← YOUR images go here
│       ├── cat/
│       ├── dog/
│       └── bird/
│
├── models/
│   └── exported/
│       └── classifier.pkl       ← trained model (auto-generated)
│
├── src/
│   ├── config_loader.py         ← typed YAML → dataclass
│   ├── data_pipeline.py         ← DataBlock, augmentation, DataLoaders
│   ├── model_builder.py         ← Learner, LR finder, export
│   ├── trainer.py               ← two-phase training loop
│   └── utils.py                 ← seed, logging, device, validation
│
├── scripts/
│   ├── download_sample_data.py  ← zero-prep dataset download
│   ├── setup_env.sh             ← one-shot setup (Linux/macOS)
│   └── setup_env.bat            ← one-shot setup (Windows)
│
├── train.py                     ← training entry point
├── predict.py                   ← inference entry point
├── evaluation.py                ← metrics report
├── api.py                       ← FastAPI REST server
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## 🚀 Quick Start

### Option A — One command (Linux / macOS)

```bash
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
chmod +x scripts/setup_env.sh && ./scripts/setup_env.sh
source .venv/bin/activate
python train.py
```

### Option B — One command (Windows)

```bat
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
scripts\setup_env.bat
.venv\Scripts\activate
python train.py
```

### Option C — Make

```bash
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
make setup      # creates venv, installs deps, downloads data
make train      # trains the model
make predict IMG=data/dataset/cat/cat_000.jpg
```

> **The setup scripts handle everything:** virtual environment, all dependencies,
> and a sample cats/dogs dataset — no manual steps required.

---

## 📦 Manual Setup (Step by Step)

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
```

### 2. Create virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

# GPU (NVIDIA CUDA)
pip install torch torchvision
pip install -r requirements.txt

# CPU only (smaller install, slower training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Prepare dataset

**Option A — Download the built-in sample dataset (cats + dogs, ~800 MB):**

```bash
python scripts/download_sample_data.py
```

**Option B — Download arbitrary classes via Bing:**

```bash
python scripts/download_sample_data.py --classes cat dog bird --limit 200
```

**Option C — Use your own images:**

```
data/dataset/
    your_class_1/   img001.jpg  img002.jpg  ...   (≥ 50 images recommended)
    your_class_2/   ...
    your_class_3/   ...
```

> Any image format works: JPEG, PNG, BMP, WebP.  
> Class name = folder name. That's the entire labelling interface.

---

## 🏋️ Training

```bash
# Standard training
python train.py

# Auto-detect best learning rate (recommended for new datasets)
python train.py --lr-finder

# Preview augmented samples before training
python train.py --show-batch

# 1-epoch smoke test (CI / debugging)
python train.py --quick

# Custom config
python train.py --config my_config.yaml
```

**Training output files:**

| File | Contents |
|---|---|
| `models/exported/classifier.pkl` | Deployable model (weights + vocab + transforms) |
| `training.log` | Full timestamped log |
| `training_history.csv` | Epoch-by-epoch loss and accuracy |
| `lr_finder_plot.png` | LR finder curve (with `--lr-finder`) |

**Expected training time:**

| Hardware | ~Time (3 classes, 150 img/class) |
|---|---|
| NVIDIA RTX 3080 | 3–6 minutes |
| NVIDIA GTX 1060 | 10–15 minutes |
| Apple M2 (MPS) | 8–12 minutes |
| CPU only | 60–120 minutes |

---

## 🔮 Prediction

```bash
# Single image — pretty output
python predict.py --image path/to/image.jpg

# Single image — JSON output
python predict.py --image path/to/image.jpg --output-format json

# Multiple images
python predict.py --images img1.jpg img2.jpg img3.jpg

# Entire folder
python predict.py --folder data/test/

# Save results to file
python predict.py --folder data/test/ --output-format json --output-file results.json

# Force CPU
python predict.py --image img.jpg --cpu

# Confidence threshold (marks uncertain predictions)
python predict.py --image img.jpg --threshold 0.7
```

**Example output:**

```
────────────────────────────────────────────────────
  Image  : data/dataset/cat/cat_042.jpg
  Result : CAT  (98.21%)

             cat: ██████████████████████████████  98.2%  ◄
             dog: ░░░░                             1.3%
            bird: ░                                0.5%
────────────────────────────────────────────────────
```

---

## 📊 Evaluation

```bash
python evaluation.py

# Custom paths
python evaluation.py --model models/exported/classifier.pkl \
                     --data  data/dataset \
                     --output eval_report.json
```

Reports: top-1 accuracy, top-3 accuracy, per-class Precision / Recall / F1, confusion matrix.

---

## ⚙️ Configuration

All hyperparameters live in `config/config.yaml` — no code changes needed:

```yaml
model:
  architecture: "resnet50"   # resnet34 | resnet50 | efficientnet_b0 | efficientnet_b3

data:
  image_size:  224           # increase to 299+ for better accuracy (more VRAM)
  batch_size:  32            # reduce to 16 if GPU runs out of memory

training:
  head_epochs:     4         # Phase 1: head-only training
  finetune_epochs: 10        # Phase 2: full fine-tuning
  mixup_alpha:     0.4       # 0 to disable MixUp
  label_smoothing: 0.1       # 0 to disable label smoothing
```

---

## 🌐 REST API

```bash
# Install API dependencies
pip install fastapi uvicorn python-multipart

# Train the model first
python train.py

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8080
```

**Endpoints:**

```bash
# Classify an image
curl -X POST http://localhost:8080/classify \
     -F "file=@cat.jpg"

# Health check
curl http://localhost:8080/health

# List classes
curl http://localhost:8080/classes
```

**JSON response:**

```json
{
  "image_path": "cat.jpg",
  "label": "cat",
  "confidence": 0.9821,
  "all_probs": {
    "cat": 0.9821,
    "dog": 0.0134,
    "bird": 0.0045
  },
  "top_3": [["cat", 0.9821], ["dog", 0.0134], ["bird", 0.0045]]
}
```

Interactive docs (Swagger UI): [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🐳 Docker

```bash
# Build
docker build -t image-classifier:latest .

# Run (mounts your local models/ directory)
docker run -p 8080:8080 \
           -v $(pwd)/models:/app/models \
           image-classifier:latest

# Test
curl http://localhost:8080/health
```

---

## 📈 Improving Accuracy

| Technique | Typical gain | How to enable |
|---|---|---|
| **More data** | +5–15% | Add images to `data/dataset/<class>/` |
| **Larger backbone** | +1–4% | `architecture: "efficientnet_b3"` in config |
| **Bigger input size** | +1–3% | `image_size: 299` in config |
| **LR finder** | +0.5–2% | `python train.py --lr-finder` |
| **Test-time augmentation** | +1–2% | See TTA section below |
| **Progressive resizing** | +1–2% | Train 128px → 224px → 320px |
| **More epochs** | varies | Increase `finetune_epochs` in config |

**Test-Time Augmentation (TTA):**

```python
# In your evaluation or predict script:
preds, targets = learn.tta(ds_idx=1)
# Averages predictions over augmented copies — free accuracy boost
```

---

## 🛠 Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Set `batch_size: 16` in config |
| `NaN loss during training` | LR too high | Use `--lr-finder` flag |
| `Accuracy < 60%` | Insufficient data | Add ≥ 200 images/class |
| `Model not found` on predict | Not trained yet | Run `python train.py` |
| `No module named fastai` | Wrong venv | `source .venv/bin/activate` |
| `FileNotFoundError: dataset` | Dataset missing | Run `python scripts/download_sample_data.py` |
| Very slow training on macOS | MPS not detected | Requires PyTorch ≥ 2.0 + macOS 12.3+ |

---

## 📋 Expected Accuracy

| Images/class | Expected top-1 accuracy (ResNet-50) |
|---|---|
| 50  | 75–85% |
| 200 | 85–92% |
| 500 | 90–95% |
| 1000+ | 93–98% |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [fast.ai](https://fast.ai) — high-level deep learning library
- [PyTorch](https://pytorch.org) — underlying tensor framework
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — sample data
- Leslie Smith — Learning Rate Range Test (2017)
