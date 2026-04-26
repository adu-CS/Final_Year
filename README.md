# Detection using Lightweight Feature Fusion Techniques

Deepfake Detection project built in **Python + PyTorch**, focused on a **lightweight baseline** using **MobileNetV3-Small** for binary classification (real vs fake). The repository includes training, evaluation (with IEEE-style reporting), dataset loading, and helper scripts for face extraction / dataset-specific evaluation.

Repo: https://github.com/adu-CS/Final_Year

---

## Project Overview

This project trains a deepfake detector on cropped face images and evaluates it using common classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC) along with **efficiency metrics** (FLOPs/Params) and **latency** (CPU/GPU).

### Key points
- Backbone: `torchvision.models.mobilenet_v3_small` (ImageNet-pretrained)
- Task: **binary classification** (Real = 0, Fake = 1)
- Loss: `BCEWithLogitsLoss`
- Input size: `224x224`
- Dataset format: folders containing image files (`.jpg/.jpeg/.png`)
- Logging: **Weights & Biases** (`wandb`)
- Efficiency: **thop** for FLOPs/parameter counting

---

## Model Details

Defined in `model.py`:
- Uses `mobilenet_v3_small(weights=DEFAULT)`
- Replaces final classifier layer with a single-unit output (`Linear(..., 1)`)
- Output is a **logit** (use `sigmoid` for probability)

## Repository Structure (high level)

- `train.py` — training loop with early stopping + cosine LR schedule + W&B logging  
- `evaluate.py` — evaluation script with optimal threshold search (by F1), latency tests, confusion matrix plot
- `model.py` — MobileNetV3-based model definition (`VanillaModel`)
- `dataset.py` — `DeepfakeDataset` that loads images from `real_dir` and `fake_dir`
- `utils.py` — seed setting, metric helpers, and FLOPs/params report via `thop`
- `extract_faces_celebdf.py`, `extract_faces_crf40.py` — helper scripts to extract faces for specific datasets
- `evaluate_celebdf.py`, `evaluate_crf40.py` — dataset-specific evaluation scripts
- Folders like `cbam/`, `mobilenetv3/`, `rppg/`, `datapipeline/` — experiment variants / modules (structure mirrors the root scripts)
