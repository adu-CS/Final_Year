import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

from model import get_model
from dataset import DeepfakeDataset
from utils import set_seed, calculate_metrics, report_efficiency

def main():
    config = {"epochs": 30, "batch_size": 16, "lr": 5e-5, "patience": 7}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project="deepfake-baseline", config=config)
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    # Paths
    train_real = r"C:\Users\srikar\Downloads\train-20260315T013946Z-3-001\train\real"
    train_fake = r"C:\Users\srikar\Downloads\train-20260315T013946Z-3-001\train\fake2"
    val_real = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\real"
    val_fake = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\fake2"

    train_loader = DataLoader(DeepfakeDataset(train_real, train_fake, True), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(DeepfakeDataset(val_real, val_fake, False), batch_size=config["batch_size"])

    model = get_model().to(DEVICE)
    
    # 1. Report Efficiency (FLOPs/Params) before training
    stats = report_efficiency(model, DEVICE)
    wandb.log(stats)

    # Freeze Backbone initially
    for param in model.spatial_features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    counter = 0

    for epoch in range(config["epochs"]):
        # Unfreeze logic at Epoch 6
        if epoch == 5:
            print("\n[INFO] Unfreezing backbone for fine-tuning...")
            for param in model.spatial_features.parameters():
                param.requires_grad = True
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-6 

        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        v_loss, y_true, y_prob = 0, [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outs = model(imgs).squeeze()
                v_loss += criterion(outs, lbls).item()
                probs = torch.sigmoid(outs)
                y_true.extend(lbls.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # Metrics
        y_pred = [1 if p > 0.5 else 0 for p in y_prob]
        acc, _, _, f1, auc = calculate_metrics(y_true, y_pred, y_prob)
        avg_v_loss = v_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} | AUC: {auc:.4f} | Loss: {avg_v_loss:.4f}")
        wandb.log({"auc": auc, "f1": f1, "val_loss": avg_v_loss, "train_loss": train_loss/len(train_loader)})
        scheduler.step()

        # Early Stopping
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            counter = 0
            torch.save(model.state_dict(), "results/vanilla_baseline.pth")
        else:
            counter += 1
            if counter >= config["patience"]: break

    wandb.finish()

if __name__ == "__main__": main()