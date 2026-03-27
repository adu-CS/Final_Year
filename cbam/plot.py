import pandas as pd
import matplotlib.pyplot as plt

# Load logs
df = pd.read_csv("results/logs.csv")

# -------- LOSS CURVE --------
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

# -------- AUC CURVE --------
plt.figure()
plt.plot(df["epoch"], df["auc"], label="AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("AUC vs Epochs")
plt.legend()
plt.show()

# -------- F1 CURVE --------
plt.figure()
plt.plot(df["epoch"], df["f1"], label="F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.title("F1 Score vs Epochs")
plt.legend()
plt.show()