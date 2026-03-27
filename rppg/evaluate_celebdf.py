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
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

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
        latency = start_event.elapsed_time(end_event) / n_runs 
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        latency = (time.perf_counter() - start) / n_runs * 1000  
    return latency

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"[INFO] Evaluating Physio (rPPG) on Celeb-DF dataset using: {DEVICE}")

    # 1. Path to your NEWly extracted Celeb-DF data
    celebdf_real = r"C:\Users\srikar\ai_engineering_hub\Deepfake_detection_project\celebdf_evaluation_data\real"
    celebdf_fake = r"C:\Users\srikar\ai_engineering_hub\Deepfake_detection_project\celebdf_evaluation_data\fake"
    
    dataset = DeepfakeDataset(celebdf_real, celebdf_fake, train=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Load the Physio Model
    checkpoint_path = "results/rppg_model.pth"
    model = get_model().to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True),
        strict=False
    )
    model.eval()

    # 3. Collect predictions
    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Collecting predictions"):
            imgs = imgs.to(DEVICE)
            outs = model(imgs).squeeze()
            if outs.dim() == 0: outs = outs.unsqueeze(0)
            y_true.extend(lbls.cpu().numpy())
            y_prob.extend(torch.sigmoid(outs).cpu().numpy())

    # 4. Benchmarking
    gpu_latency = measure_latency(model, DEVICE)
    model.to("cpu")
    cpu_latency = measure_latency(model, torch.device("cpu"))
    model.to(DEVICE) 

    stats = report_efficiency(model.to("cpu"), torch.device("cpu"))
    param_count = stats["params_M"]
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # 5. Metric Calculations
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    thresholds = np.linspace(0, 1, 500)
    f1_scores = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob > best_t).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # 6. IEEE Report Output
    print("\n" + "=" * 54)
    print("      IEEE GENERALIZATION REPORT — PHYSIO MODEL (CELEB-DF)")
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

    # 7. Save Plots
    os.makedirs("results/celebdf", exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Purples',
                xticklabels=['Pred Real', 'Pred Fake'], yticklabels=['Actual Real', 'Actual Fake'])
    plt.title(f'Physio (Celeb-DF) — MCC: {mcc:.4f}')
    plt.tight_layout()
    plt.savefig("results/celebdf/physio_confusion_matrix.png", dpi=150)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='rebeccapurple', label=f'Physio AUC={auc:.4f}')
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.title('ROC Curve — Physio (Celeb-DF Generalization)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/celebdf/physio_roc_curve.png", dpi=150)
    plt.close('all')

    print("\n[SUCCESS] Celeb-DF generalization results for Physio saved to results/celebdf/")

if __name__ == "__main__":
    evaluate()