import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import wandb

from model import get_model
from dataset import DeepfakeDataset
from utils import set_seed, calculate_metrics, report_efficiency


class LabelSmoothingBCELoss(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing to prevent overconfidence.
    """
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits.squeeze(), targets_smooth)


def main():
    config = {
        "epochs":        200,   
        "batch_size":    16,
        "lr":            5e-5,
        "weight_decay":  1e-4,  
        "dropout":       0.2,   
        "label_smooth":  0.05,  
        "patience":      15,    
        "unfreeze_epoch": 5,    
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

    wandb.init(project="deepfake-rppg", config=config)
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    train_real = r"C:\Users\srikar\Downloads\train-20260315T013946Z-3-001\train\real"
    train_fake = r"C:\Users\srikar\Downloads\train-20260315T013946Z-3-001\train\fake2"
    val_real   = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\real"
    val_fake   = r"C:\Users\srikar\Downloads\val-20260315T013948Z-3-001\val\fake2"

    train_dataset = DeepfakeDataset(train_real, train_fake, train=True)
    val_dataset   = DeepfakeDataset(val_real,   val_fake,   train=False)

    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        sampler=sampler,        
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Model Setup
    # ------------------------------------------------------------------ #
    model = get_model(dropout_rate=config["dropout"]).to(DEVICE)
    stats = report_efficiency(model, DEVICE)
    wandb.log(stats)

    # Freeze spatial backbone initially; Physio branch trains from scratch
    for param in model.spatial_features.parameters():
        param.requires_grad = False
    print("[INFO] Backbone frozen — training Physio Branch + Classifier head only.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=1e-7,
    )

    criterion = LabelSmoothingBCELoss(smoothing=config["label_smooth"])

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(config["epochs"]):

        if epoch == config["unfreeze_epoch"]:
            print(f"\n[INFO] Epoch {epoch+1}: Unfreezing spatial backbone for fine-tuning...")
            for param in model.spatial_features.parameters():
                param.requires_grad = True
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-6

        model.train()
        train_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} | Train", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        y_true, y_prob = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outs = model(imgs).squeeze()
                if outs.dim() == 0:
                    outs = outs.unsqueeze(0)
                val_loss += criterion(outs, lbls).item()
                y_true.extend(lbls.cpu().numpy())
                y_prob.extend(torch.sigmoid(outs).cpu().numpy())

        y_pred = [1 if p > 0.5 else 0 for p in y_prob]
        acc, prec, recall, f1, auc = calculate_metrics(y_true, y_pred, y_prob)
        avg_val_loss   = val_loss   / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch+1:03d} | "
            f"AUC: {auc:.4f} | F1: {f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Train Loss: {avg_train_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        wandb.log({
            "epoch":       epoch + 1,
            "auc":         auc,
            "f1":          f1,
            "accuracy":    acc,
            "precision":   prec,
            "recall":      recall,
            "val_loss":    avg_val_loss,
            "train_loss":  avg_train_loss,
            "lr":          optimizer.param_groups[0]['lr'],
        })

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "results/rppg_model.pth")
            print(f"  [Checkpoint] Saved best model (val_loss={best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  [EarlyStopping] {early_stop_counter}/{config['patience']} "
                  f"(best val_loss={best_val_loss:.4f})")
            if early_stop_counter >= config["patience"]:
                print(f"\n[STOP] Early stopping triggered at epoch {epoch+1}.")
                break

    wandb.finish()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print("Best model saved to: results/rppg_model.pth")


if __name__ == "__main__":
    main()