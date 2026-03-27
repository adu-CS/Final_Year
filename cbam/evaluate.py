import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             f1_score, roc_auc_score, accuracy_score, 
                             precision_score, recall_score, matthews_corrcoef)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from dataset import DeepfakeDataset
from utils import set_seed

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 1. Load Data
    val_real = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\real"
    val_fake = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\fake2"
    dataset = DeepfakeDataset(val_real, val_fake, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # BS=1 for latency testing

    # 2. Load Model
    model = get_model().to(DEVICE)
    checkpoint_path = "results/vanilla_baseline.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
    model.eval()

    y_true, y_prob = [], []

    # 3. Inference & Latency (GPU)
    print(f"Testing Inference on {DEVICE}...")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Warm-up
    for _ in range(10): _ = model(dummy_input)
    
    start_gpu = time.time()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Collecting Predictions"):
            imgs = imgs.to(DEVICE)
            outs = model(imgs).squeeze()
            if outs.dim() == 0: outs = outs.unsqueeze(0)
            probs = torch.sigmoid(outs)
            y_true.extend(lbls.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    end_gpu = time.time()
    
    gpu_latency = (end_gpu - start_gpu) / len(dataset) * 1000 # ms per image

    # 4. Latency (CPU)
    model.to("cpu")
    dummy_cpu = torch.randn(1, 3, 224, 224)
    start_cpu = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_cpu)
    end_cpu = time.time()
    cpu_latency = (end_cpu - start_cpu) / 100 * 1000 # ms per image

    # 5. Model Stats
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # 6. Metrics Calculation
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    thresholds = np.linspace(0, 1, 100)
    f1_list = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1_list)]
    
    y_pred = (y_prob > best_t).astype(int)
    
    # Core IEEE Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred) # Sensitivity
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # 7. Final Report
    print("\n" + "="*50)
    print("         IEEE FINAL PERFORMANCE REPORT")
    print("="*50)
    print(f"{'Metric':<25} | {'Value':<20}")
    print("-" * 50)
    print(f"{'Optimal Threshold':<25} | {best_t:.4f}")
    print(f"{'Accuracy':<25} | {acc:.4f}")
    print(f"{'Precision':<25} | {prec:.4f}")
    print(f"{'Recall (Sensitivity)':<25} | {recall:.4f}")
    print(f"{'Specificity':<25} | {specificity:.4f}")
    print(f"{'F1-Score':<25} | {f1:.4f}")
    print(f"{'MCC':<25} | {mcc:.4f}")
    print(f"{'ROC-AUC':<25} | {auc:.4f}")
    print("-" * 50)
    print(f"{'Latency (CPU)':<25} | {cpu_latency:.2f} ms/img")
    print(f"{'Latency (GPU)':<25} | {gpu_latency:.2f} ms/img")
    print(f"{'Parameters':<25} | {param_count:.2f} Million")
    print(f"{'Model Size':<25} | {model_size_mb:.2f} MB")
    print("="*50)

    # Plotting Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix (MCC: {mcc:.3f})')
    plt.savefig("results/ieee_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()