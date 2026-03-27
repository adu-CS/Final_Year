import torch
import numpy as np
import random
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)

try:
    from thop import profile
except ImportError:
    print("[WARNING] thop not installed. Run: pip install thop")
    profile = None


def set_seed(seed=42):
    """
    Fix all random seeds for full reproducibility.
    IEEE mandate: reviewers must be able to replicate results exactly.
    Covers Python random, NumPy, PyTorch CPU and GPU, and cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Computes core classification metrics.
    zero_division=0 silences warnings when a class has no predictions —
    common in early training epochs before the model has calibrated.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Raised if only one class is present in y_true (early training edge case)
        auc = 0.5

    return acc, precision, recall, f1, auc


def report_efficiency(model, device, input_size=(1, 3, 224, 224)):
    """
    Computes IEEE complexity metrics: FLOPs and parameter count.
    Called once before training and logged to W&B.

    FLOPs (floating point operations) measure computational complexity.
    MACs (multiply-accumulate operations) = FLOPs / 2 — thop reports MACs,
    commonly called FLOPs in the literature. Both are reported here.
    """
    if profile is None:
        print("[WARNING] Skipping efficiency report — thop not installed.")
        return {"params_M": 0.0, "flops_G": 0.0, "macs_G": 0.0}

    model.eval()
    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)

    params_M = params / 1e6
    macs_G   = macs   / 1e9
    flops_G  = macs_G * 2   # FLOPs = 2 * MACs (each MAC = 1 mul + 1 add)

    print(f"\n{'=' * 38}")
    print(f"  IEEE EFFICIENCY REPORT — MODEL")
    print(f"{'=' * 38}")
    print(f"  Parameters : {params_M:.4f} Million")
    print(f"  MACs       : {macs_G:.4f}  GMACs")
    print(f"  FLOPs      : {flops_G:.4f} GFLOPs")
    print(f"{'=' * 38}\n")

    return {"params_M": params_M, "macs_G": macs_G, "flops_G": flops_G}