import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    from thop import profile
except ImportError:
    print("Please install thop: pip install thop")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    return acc, precision, recall, f1, auc

def report_efficiency(model, device, input_size=(1, 3, 224, 224)):
    """Calculates IEEE complexity metrics: FLOPs and Params."""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    print(f"\n{'='*30}")
    print(f"IEEE EFFICIENCY REPORT")
    print(f"{'='*30}")
    print(f"Total Parameters: {params/1e6:.2f} Million")
    print(f"Total FLOPs: {flops/1e9:.3f} GFLOPs")
    print(f"{'='*30}\n")
    
    return {"params_M": params/1e6, "flops_G": flops/1e9}