# -*- coding: utf-8 -*-
"""CINIC-10 MLP Pipeline (Colab-ready)

Paste this script into Google Colab as alternating Markdown and Code cells.
Install deps in a separate Colab cell:
  !pip install -q torch torchvision tensorboard matplotlib scikit-learn seaborn wandb
"""

# Colab guard (safe outside Colab)
try:
    from google.colab import drive, userdata  # type: ignore
    drive.mount('/content/drive')
except Exception:
    drive = None
    userdata = None

# (If running this .py outside Colab, comment out the pip lines above.)

import os, random, json, copy, math
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

"""# Setup and Paths

Config values and output directories. Adjust `FOLDER_PATH` to your CINIC-10 root.
"""
# -------------------------
# Config / Paths
# -------------------------
wandb_project = os.environ.get("WANDB_PROJECT", "cinic10-mlp")

# Point directly to cinic-10
FOLDER_PATH = "/content/drive/My Drive/cinic-10"
DATA_DIR = Path(FOLDER_PATH)
OUT_DIR = Path("/content/drive/My Drive/Masters in Intelligent Computing Systems/Deep Learning/Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("DATA_DIR:", DATA_DIR)
print("Train exists?", (DATA_DIR / "train").exists())

# W&B helper: only log if a run is active
def wandb_log_safe(data: dict):
    try:
        if wandb.run is not None:
            wandb.log(data)
    except Exception as e:
        print(f"W&B logging skipped: {e}")

"""# 1) Data Exploration

Quickly inspect the CINIC-10 directory structure and counts per split/class.
"""
# -------------------------
# 1) Data exploration (quick checks)
# -------------------------
def explore_dataset(base_dir):
    info = {}
    for split in ["train", "valid", "test"]:
        p = Path(base_dir) / split
        if not p.exists():
            info[split] = {"exists": False}
            continue
        classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
        counts = {c: len(list((p/c).glob("*"))) for c in classes}
        total = sum(counts.values())
        info[split] = {"exists": True, "total_images": total, "num_classes": len(classes), "per_class": counts, "classes": classes}
    return info

expl = explore_dataset(DATA_DIR)
print("Dataset exploration summary:")
print(json.dumps(expl, indent=2)[:2000])  # truncated print

# Known dataset facts (CINIC-10): 270,000 images; 90k per split; 32x32; 3 channels (RGB).
# If your local dataset differs, expl above will show counts.

"""# 2) Preprocessing & Augmentations

Normalization (CINIC-10 statistics) and light augmentations for training.
"""
# -------------------------
# 2) Preprocessing & augmentations
# -------------------------
# CINIC-10 recommended normalization (common practice). Adjust if desired.
MEAN = [0.47889522, 0.47227842, 0.43047404]
STD  = [0.24205776, 0.23828046, 0.25874835]

train_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load ImageFolder datasets (expects directory/classname structure)
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform_aug)
valid_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=eval_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)

print("Loaded datasets sizes -> train:", len(train_ds), "valid:", len(valid_ds), "test:", len(test_ds))
CLASS_NAMES = train_ds.classes

"""# 2b) 30% Small Subset

Create a 30% random subset of the training data and split it equally into
train/eval/test slices (~10% each) for quick experimentation and grid search.
"""
# -------------------------
# 2b) Create small subset: randomly select 30% of training data (will be split into three equal parts = 10% each)
# -------------------------
SMALL_SUBSET_FRAC = 0.30
small_n = int(len(train_ds) * SMALL_SUBSET_FRAC)
all_indices = list(range(len(train_ds)))
random.seed(42)
random.shuffle(all_indices)
small_indices = all_indices[:small_n]
small_subset = Subset(train_ds, small_indices)
print("Small subset size (30% of train):", len(small_subset))

# Split small_subset into three equal parts (each ~10% of original train)
third = len(small_subset) // 3
lens = [third, third, len(small_subset) - 2*third]
small_train_subset, small_eval_subset, small_test_subset = random_split(small_subset, lens)
print("Small splits:", len(small_train_subset), len(small_eval_subset), len(small_test_subset))

# Convenience: function to get x_train,y_train arrays if needed
def subset_to_xy(subset):
    # Returns numpy arrays (X flattened not returned here because images are stored in dataset)
    X_idx = [subset.indices[i] if isinstance(subset, Subset) else i for i in range(len(subset))]
    y = [subset.dataset.samples[idx][1] if isinstance(subset, Subset) else subset.dataset.samples[idx][1] for idx in X_idx]
    return X_idx, y

"""# 3) DataLoaders

Helpers to build DataLoader objects for the small subset and the final combined set.
"""
# -------------------------
# 3) DataLoaders: helper
# -------------------------
def make_dataloaders(batch_size, use_aug_on_full_train=False):
    # For small subset (used in grid search)
    train_loader = DataLoader(small_train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(small_eval_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(small_test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, eval_loader, test_loader

# Full train loaders (for final training)
def make_full_train_loader(batch_size, augment=True):
    if augment:
        # re-create ImageFolder with augmentation for both train and valid to increase diversity on final combined training
        train_aug = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform_aug)
        valid_aug = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=train_transform_aug)
        combined = ConcatDataset([train_aug, valid_aug])
    else:
        combined = ConcatDataset([train_ds, valid_ds])
    return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

test_loader_full = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
valid_loader_full = DataLoader(valid_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

"""# 4) Model: Feedforward MLP

Fully-connected network with ReLU and Dropout. Output layer is logits for 10 classes.
CrossEntropyLoss applies softmax during loss computation.
"""
# -------------------------
# 4) Model: L-layer feedforward network (MLP)
# -------------------------
class FeedForwardMLP(nn.Module):
    def __init__(self, input_size=3*32*32, hidden_sizes=[2048, 1024, 512, 256], num_classes=10, dropout=0.5):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))  # logits (CrossEntropyLoss applies softmax)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)

"""# Regularization

L1 penalty helper (in addition to L2 via weight decay in the optimizer).
"""
# -------------------------
# Regularization helpers (L1)
# -------------------------
def l1_penalty(model):
    l1 = torch.tensor(0., device=device)
    for p in model.parameters():
        l1 += torch.norm(p, 1)
    return l1

"""# 5) Training and Evaluation Utilities

Training loop with optional L1, early stopping (by patience), TensorBoard logging,
and W&B metric logging. `evaluate()` computes and logs metrics and confusion matrices.
"""
# -------------------------
# Training / Eval loops + Early Stopping + TensorBoard logging
# -------------------------
def train_epoch(model, optimizer, criterion, dataloader, l1_lambda=0.0):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for X, y in dataloader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_penalty(model)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc

def evaluate(model, criterion, dataloader, class_names=None, log_to_wandb=True, prefix="val"):
    """
    Evaluate the model on a dataloader.
    Logs metrics (loss, accuracy, precision, recall, F1) to W&B if enabled.

    Args:
        model: trained PyTorch model
        criterion: loss function
        dataloader: DataLoader to evaluate on
        class_names: list of class names (optional, for confusion matrix)
        log_to_wandb: whether to log metrics to W&B
        prefix: "val" or "test" (used as key prefix in logs)
    """
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        f"{prefix}_loss": avg_loss,
        f"{prefix}_accuracy": acc,
        f"{prefix}_precision": prec,
        f"{prefix}_recall": rec,
        f"{prefix}_f1": f1
    }

    # Log to W&B (safely)
    if log_to_wandb:
        wandb_log_safe(metrics)

        # Confusion matrix
        if class_names is not None:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
            wandb_log_safe({f"{prefix}_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

    return avg_loss, acc, prec, rec, f1, y_true, y_pred

class EarlyStopper:
    def __init__(self, patience=7, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
    def step(self, value):
        if self.best is None:
            self.best = value; return False
        improvement = value - self.best if self.mode=="max" else self.best - value
        if improvement > self.min_delta:
            self.best = value; self.counter = 0; return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def run_training(model, train_loader, val_loader, cfg, writer=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        # --------------------
        # Train
        # --------------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if cfg.get("l1_lambda", 0.0) > 0.0:
                loss = loss + cfg["l1_lambda"] * l1_penalty(model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            _, predicted = preds.max(1)
            correct += predicted.eq(yb).sum().item()
            total += yb.size(0)
        train_loss /= total
        train_acc = correct / total

        # --------------------
        # Validate
        # --------------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                _, predicted = preds.max(1)
                correct += predicted.eq(yb).sum().item()
                total += yb.size(0)
        val_loss /= total
        val_acc = correct / total

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        lr_curr = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr_curr)

        # --------------------
        # Logging
        # --------------------
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", lr_curr, epoch)

        wandb_log_safe({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": lr_curr,
        })

        print(f"Epoch {epoch+1}/{cfg['epochs']} "
              f"Train loss={train_loss:.4f}, acc={train_acc:.4f} "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        # --------------------
        # Early stopping
        # --------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model)
    return model, history

"""# 6) Grid Search

Brute-force search over learning rate, weight decay, and batch size on the 30% subset
with small epochs for speed. Selects the best by validation accuracy.
"""
# -------------------------
# 5) Grid Search (lr, weight_decay, batch_size)
#    Quick & conservative (small epochs). We'll keep epochs small to save time for grid search.
# -------------------------
param_grid = {
    "lr": [0.1, 0.01, 0.001],
    "weight_decay": [5e-4, 1e-4, 0.0],
    "batch_size": [64, 128]
}
grid_results = []
grid_config = {"epochs": 12, "momentum": 0.9, "l1_lambda": 0.0, "patience": 4}

print("Starting grid search on the 30% small subset (this will iterate over combinations).")
for lr in param_grid["lr"]:
    for wd in param_grid["weight_decay"]:
        for bs in param_grid["batch_size"]:
            cfg = {"lr": lr, "weight_decay": wd, "momentum": grid_config["momentum"],
                   "epochs": grid_config["epochs"], "l1_lambda": grid_config["l1_lambda"], "patience": grid_config["patience"]}
            train_loader, val_loader, _ = make_dataloaders(bs)
            run_name = f"grid_lr{lr}_wd{wd}_bs{bs}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", run_name))
            model = FeedForwardMLP(hidden_sizes=[2048,1024,512,256], dropout=0.5)
            model, history = run_training(model, train_loader, val_loader, cfg, writer=writer)
            best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
            grid_results.append({"params": {"lr": lr, "weight_decay": wd, "batch_size": bs}, "val_acc": best_val_acc, "history": history})
            writer.close()
            print(f"Grid combo lr={lr},wd={wd},bs={bs} -> best_val_acc={best_val_acc:.4f}")

# choose best
best_entry = max(grid_results, key=lambda x: x["val_acc"])
best_params = best_entry["params"]
print("GRID SEARCH COMPLETE. Best params:", best_params, "val_acc:", best_entry["val_acc"])

# save best hyperparams
with open(os.path.join(OUT_DIR, "best_hyperparams.json"), "w") as f:
    json.dump(best_params, f)

# Optional W&B login via Colab userdata secret (set in Colab > User data)
if 'userdata' in globals() and userdata is not None:
    try:
        api_key = userdata.get('wandb_api')
        if api_key:
            wandb.login(key=api_key)
    except Exception as e:
        print("W&B login skipped:", e)

"""# 7) Train with Best Params (Small Subset)

Trains on the small subset using best hyperparameters; logs to W&B and evaluates on
the small test slice.
"""
# -------------------------
# 6) Train with best params + log to W&B
# -------------------------
bs = best_params["batch_size"]
train_loader, val_loader, small_test_loader = make_dataloaders(bs)

wandb.init(
    project=wandb_project,
    name=f"best_small_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    config={
        "batch_size": bs,
        "lr": best_params["lr"],
        "weight_decay": best_params["weight_decay"],
        "momentum": 0.9,
        "epochs": 20,
        "l1_lambda": 0.0,
        "patience": 6,
        "model": "FeedForwardMLP",
        "hidden_sizes": [2048,1024,512,256],
        "dropout": 0.5
    }
)

model = FeedForwardMLP(hidden_sizes=[2048,1024,512,256], dropout=0.5)
cfg = wandb.config
model, history = run_training(model, train_loader, val_loader, cfg, writer=None)

# Evaluate on small test set (the 3rd slice of small subset)
criterion = nn.CrossEntropyLoss()
val_loss, val_acc, val_prec, val_rec, val_f1, y_true_small, y_pred_small = evaluate(
    model, criterion, small_test_loader, class_names=CLASS_NAMES, prefix="small_test"
)
print("Small test set results -> acc:", val_acc, "prec:", val_prec, "rec:", val_rec, "f1:", val_f1)
print("Confusion matrix (small test):")
print(confusion_matrix(y_true_small, y_pred_small))

# Save small-test model
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_small_best.pth"))

# Record small-test metrics in W&B summary
if wandb.run is not None:
    try:
        wandb.run.summary.update({
            "small_test_loss": val_loss,
            "small_test_accuracy": val_acc,
            "small_test_precision": val_prec,
            "small_test_recall": val_rec,
            "small_test_f1": val_f1,
        })
    except Exception as e:
        print("W&B summary update skipped:", e)

wandb.finish()

"""# 8) Final Training on Combined Train+Valid

Combine train and valid (optionally with augmentation) and train the final model.
Validate on the original valid set and test on the original test split.
"""
# -------------------------
# 7) After success: Combine train + valid (augment both) for final training; evaluate on held-out test (original test split)
# -------------------------
print("Final training on combined train+valid (with augmentation) using best hyperparams.")
final_train_loader = make_full_train_loader(batch_size=best_params["batch_size"], augment=True)
final_cfg = {"lr": best_params["lr"], "weight_decay": best_params["weight_decay"], "momentum": 0.9, "epochs": 50, "l1_lambda": 0.0, "patience": 8}
tb_name = f"final_combined_{datetime.now().strftime('%Y%m%d%H%M%S')}"
writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", tb_name))
wandb.init(
    project=wandb_project,
    name=tb_name,
    config={
        **final_cfg,
        "batch_size": best_params["batch_size"],
        "model": "FeedForwardMLP",
        "hidden_sizes": [2048,1024,512,256],
        "dropout": 0.5,
        "phase": "final_combined"
    }
)

final_model = FeedForwardMLP(hidden_sizes=[2048,1024,512,256], dropout=0.5)
final_model, final_history = run_training(final_model, final_train_loader, valid_loader_full, final_cfg, writer=writer)
writer.close()

# Evaluate on held-out test (original test split)
criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_prec, test_rec, test_f1, y_true_test, y_pred_test = evaluate(
    final_model, criterion, test_loader_full, class_names=CLASS_NAMES, prefix="test"
)
print("Held-out TEST set metrics -> acc:%.4f prec:%.4f rec:%.4f f1:%.4f" % (test_acc, test_prec, test_rec, test_f1))
print("Confusion matrix (test):")
print(confusion_matrix(y_true_test, y_pred_test))

# Record final test metrics in W&B summary
if wandb.run is not None:
    try:
        wandb.run.summary.update({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
        })
    except Exception as e:
        print("W&B summary update skipped:", e)



# Save final model and summary
torch.save(final_model.state_dict(), os.path.join(OUT_DIR, "final_model_combined.pth"))
summary = {
    "best_hyperparams": best_params,
    "small_test_metrics": {"acc": val_acc, "prec": val_prec, "rec": val_rec, "f1": val_f1},
    "final_test_metrics": {"acc": test_acc, "prec": test_prec, "rec": test_rec, "f1": test_f1}
}
with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

"""# 9) Final Metrics & Plots

Plot training curves (loss/acc/lr) and print a classification report.
"""
# -------------------------
# 8) Final metrics & plots (loss/acc/LR) saved & shown
# -------------------------
def plot_history(hist, title_prefix="history", save_path=None):
    epochs = range(1, len(hist["train_loss"])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(epochs, hist["train_acc"], label="train_acc")
    plt.plot(epochs, hist["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(epochs, hist["lr"], label="lr")
    plt.title("Learning Rate")
    plt.legend()
    plt.suptitle(title_prefix)
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_history(final_history, title_prefix="Final Combined Training History", save_path=os.path.join(OUT_DIR, "final_training_history.png"))

# Display final classification report on held-out test
print("FINAL CLASSIFICATION REPORT (held-out test):")
print(classification_report(y_true_test, y_pred_test, digits=4))

print("Artifacts saved to:", OUT_DIR)
print("To inspect TensorBoard in Colab run:")
print("  %load_ext tensorboard")
print(f"  %tensorboard --logdir {os.path.join(OUT_DIR,'runs')}")

wandb.finish()

# Done.

"""Dont Run"""

# -------------------------
# 8) (Optional) Train with FULL training set (train+valid+test) IF you want a final model for production.
#    NOTE: training on test removes your held-out evaluation. Save separately if you do this.
# -------------------------
# If you want to create a "production" model trained on all labeled data:

"""
full_all = ConcatDataset([datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform_aug),
                          datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=train_transform_aug),
                          datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=train_transform_aug)])
full_all_loader = DataLoader(full_all, batch_size=best_params["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

prod_model = FeedForwardMLP(hidden_sizes=[2048,1024,512,256], dropout=0.5).to(device)
prod_cfg = {"lr": best_params["lr"], "weight_decay": best_params["weight_decay"], "momentum": 0.9, "epochs": 30, "l1_lambda": 0.0, "patience": 6}
tb_name = f"prod_all_{datetime.now().strftime('%Y%m%d%H%M%S')}"
writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "runs", tb_name))
prod_model, prod_hist = run_training(prod_model, full_all_loader, valid_loader_full, prod_cfg, writer=writer)
writer.close()
torch.save(prod_model.state_dict(), os.path.join(OUT_DIR, "prod_model_all_data.pth"))

"""

