import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import (
    confusion_matrix, roc_curve,
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, matthews_corrcoef,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from dataset import DeepfakeDataset
from utils import set_seed, report_efficiency


def measure_latency(model, device, n_warmup=50, n_runs=200):
    """
    Measures pure model inference latency using dummy tensors.
    """
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event) / n_runs  # ms
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        latency = (time.perf_counter() - start) / n_runs * 1000  # ms

    return latency


def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"[INFO] Evaluating on: {DEVICE}")

    # ------------------------------------------------------------------ #
    # 1. Validation data
    # ------------------------------------------------------------------ #
    val_real = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\real"
    val_fake = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\fake2"
    dataset  = DeepfakeDataset(val_real, val_fake, train=False)
    loader   = DataLoader(dataset, batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------ #
    # 2. Model Loading (with strict=False bypass for thop artifacts)
    # ------------------------------------------------------------------ #
    checkpoint_path = "results/rppg_model.pth"
    model = get_model().to(DEVICE)
    
    # The fix: strict=False forces PyTorch to ignore the 'total_ops' keys
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True),
        strict=False
    )
    model.eval()

    # ------------------------------------------------------------------ #
    # 3. Collect predictions
    # ------------------------------------------------------------------ #
    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Collecting predictions"):
            imgs = imgs.to(DEVICE)
            outs = model(imgs).squeeze()
            if outs.dim() == 0:
                outs = outs.unsqueeze(0)
            y_true.extend(lbls.cpu().numpy())
            y_prob.extend(torch.sigmoid(outs).cpu().numpy())

    # ------------------------------------------------------------------ #
    # 4. Latency
    # ------------------------------------------------------------------ #
    gpu_latency = measure_latency(model, DEVICE)
    model.to("cpu")
    cpu_latency = measure_latency(model, torch.device("cpu"))
    model.to(DEVICE)  

    # ------------------------------------------------------------------ #
    # 5. Model complexity
    # ------------------------------------------------------------------ #
    stats          = report_efficiency(model.to("cpu"), torch.device("cpu"))
    param_count    = stats["params_M"]
    model_size_mb  = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # ------------------------------------------------------------------ #
    # 6. Metrics 
    # ------------------------------------------------------------------ #
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = np.linspace(0, 1, 500)
    f1_scores  = [
        f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob > best_t).astype(int)

    acc         = accuracy_score(y_true, y_pred)
    prec        = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    auc         = roc_auc_score(y_true, y_prob)
    mcc         = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # ------------------------------------------------------------------ #
    # 7. IEEE performance report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 54)
    print("      IEEE FINAL PERFORMANCE REPORT — RPPG MODEL")
    print("=" * 54)
    print(f"{'Metric':<28} | {'Value'}")
    print("-" * 54)
    print(f"{'Optimal threshold':<28} | {best_t:.4f}")
    print(f"{'Accuracy':<28} | {acc:.4f}")
    print(f"{'Precision':<28} | {prec:.4f}")
    print(f"{'Recall (sensitivity)':<28} | {recall:.4f}")
    print(f"{'Specificity':<28} | {specificity:.4f}")
    print(f"{'F1-score':<28} | {f1:.4f}")
    print(f"{'MCC':<28} | {mcc:.4f}")
    print(f"{'ROC-AUC':<28} | {auc:.4f}")
    print("-" * 54)
    print(f"{'Latency (GPU, ms/img)':<28} | {gpu_latency:.2f}")
    print(f"{'Latency (CPU, ms/img)':<28} | {cpu_latency:.2f}")
    print(f"{'Parameters (M)':<28} | {param_count:.3f}")
    print(f"{'Model size (MB)':<28} | {model_size_mb:.2f}")
    print("=" * 54)

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------ #
    # 8. Confusion matrix
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        [[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Real', 'Predicted Fake'],
        yticklabels=['Actual Real',    'Actual Fake'],
    )
    plt.title(f'MobileNetV3 + Physio Branch — Confusion Matrix\nMCC: {mcc:.4f}  |  AUC: {auc:.4f}')
    plt.tight_layout()
    plt.savefig("results/rppg_confusion_matrix.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    # 9. ROC curve
    # ------------------------------------------------------------------ #
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'MobileNetV3 + Physio  AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve — MobileNetV3 + Physiological Branch')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("results/rppg_roc_curve.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    # 10. F1 vs threshold curve
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1_scores, color='steelblue', lw=1.5)
    plt.axvline(best_t, color='red', linestyle='--', lw=1,
                label=f'Optimal threshold = {best_t:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.title('F1-score vs Classification Threshold — Physio Branch')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rppg_f1_threshold.png", dpi=150)
    plt.close()

    print("\n[SUCCESS] All plots saved to results/")


if __name__ == "__main__":
    evaluate()