#!/usr/bin/env python3
"""
CINIC-10 Feedforward MLP Pipeline

This script implements a comprehensive machine learning pipeline for image classification
on the CINIC-10 dataset using a feedforward multilayer perceptron (MLP). The pipeline
includes data exploration, preprocessing, hyperparameter tuning, training, and evaluation.

Key Features:
- Data exploration and validation
- Image preprocessing with normalization and augmentation
- Grid search for hyperparameter optimization
- Training with early stopping and regularization
- Comprehensive evaluation with metrics and visualizations
- TensorBoard and Weights & Biases logging
- Model persistence and experiment tracking

Dataset: CINIC-10 (32x32 RGB images, 10 classes, ~270k images)
Architecture: Feedforward MLP with configurable hidden layers
Regularization: Dropout, L1/L2 penalties, early stopping

Author: Based on the Model Design Document
Date: 2025
"""

import os
import random
import json
import copy
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import wandb

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# Scikit-learn imports for metrics
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def setup_environment():
    """
    Initialize the environment, paths, and device configuration.
    
    Returns:
        tuple: (device, data_dir, out_dir, wandb_project)
    """
    # Project configuration
    wandb_project = os.environ.get("WANDB_PROJECT", "cinic10-mlp")
    
    # Directory paths - adjust FOLDER_PATH to your CINIC-10 dataset location
    FOLDER_PATH = "./data"
    data_dir = Path(FOLDER_PATH)
    out_dir = Path("./Output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration - use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Train directory exists: {(data_dir / 'train').exists()}")
    
    return device, data_dir, out_dir, wandb_project

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wandb_log_safe(data: dict):
    """
    Safely log data to Weights & Biases, handling cases where no run is active.
    
    Args:
        data (dict): Dictionary of metrics to log
    """
    try:
        if wandb.run is not None:
            wandb.log(data)
    except Exception as e:
        print(f"W&B logging skipped: {e}")

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA EXPLORATION AND PREPROCESSING
# ============================================================================

def explore_dataset(base_dir):
    """
    Explore the CINIC-10 dataset structure and provide summary statistics.
    
    This function scans the directory structure to count classes and per-class 
    image counts for each split (train/valid/test). It validates dataset 
    integrity and provides a quick summary.
    
    Args:
        base_dir (str or Path): Path to the CINIC-10 dataset root directory
        
    Returns:
        dict: Dictionary containing dataset information for each split
    """
    info = {}
    
    for split in ["train", "valid", "test"]:
        split_path = Path(base_dir) / split
        
        if not split_path.exists():
            info[split] = {"exists": False}
            continue
            
        # Get class directories (should be 10 for CINIC-10)
        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        
        # Count images per class
        counts = {}
        for class_name in classes:
            class_path = split_path / class_name
            image_count = len(list(class_path.glob("*")))
            counts[class_name] = image_count
        
        total_images = sum(counts.values())
        
        info[split] = {
            "exists": True,
            "total_images": total_images,
            "num_classes": len(classes),
            "per_class": counts,
            "classes": classes
        }
    
    return info

def setup_data_transforms():
    """
    Create data transformation pipelines for training and evaluation.
    
    Training transforms include augmentation (horizontal flip, rotation, color jitter)
    to improve generalization. Evaluation transforms only include normalization.
    
    CINIC-10 normalization statistics:
    - MEAN = [0.47889522, 0.47227842, 0.43047404]
    - STD = [0.24205776, 0.23828046, 0.25874835]
    
    Returns:
        tuple: (train_transform_aug, eval_transform, mean, std)
    """
    # CINIC-10 dataset statistics for normalization
    MEAN = [0.47889522, 0.47227842, 0.43047404]
    STD = [0.24205776, 0.23828046, 0.25874835]
    
    # Training transforms with augmentation
    train_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomRotation(15),           # Random rotation ±15 degrees
        transforms.ColorJitter(                  # Color augmentation
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.02
        ),
        transforms.ToTensor(),                   # Convert to tensor
        transforms.Normalize(MEAN, STD),         # Normalize using dataset statistics
    ])
    
    # Evaluation transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    return train_transform_aug, eval_transform, MEAN, STD

def load_datasets(data_dir, train_transform, eval_transform):
    """
    Load CINIC-10 datasets using ImageFolder.
    
    Args:
        data_dir (Path): Path to dataset directory
        train_transform: Transform pipeline for training data
        eval_transform: Transform pipeline for validation/test data
        
    Returns:
        tuple: (train_ds, valid_ds, test_ds, class_names)
    """
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), 
        transform=train_transform
    )
    valid_ds = datasets.ImageFolder(
        os.path.join(data_dir, "valid"), 
        transform=eval_transform
    )
    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, "test"), 
        transform=eval_transform
    )
    
    class_names = train_ds.classes
    
    print(f"Dataset sizes -> train: {len(train_ds)}, valid: {len(valid_ds)}, test: {len(test_ds)}")
    print(f"Classes: {class_names}")
    
    return train_ds, valid_ds, test_ds, class_names

def create_small_subset(train_ds, subset_fraction=0.30, seed=42):
    """
    Create a small subset of the training data for rapid experimentation.
    
    This function creates a random subset of the training data and splits it
    into three equal parts for quick grid search and model iteration.
    
    Args:
        train_ds: Training dataset
        subset_fraction (float): Fraction of training data to use (default: 0.30)
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (small_train_subset, small_eval_subset, small_test_subset)
    """
    # Calculate subset size
    small_n = int(len(train_ds) * subset_fraction)
    
    # Create random indices
    all_indices = list(range(len(train_ds)))
    random.seed(seed)
    random.shuffle(all_indices)
    small_indices = all_indices[:small_n]
    
    # Create subset
    small_subset = Subset(train_ds, small_indices)
    print(f"Small subset size ({subset_fraction*100}% of train): {len(small_subset)}")
    
    # Split into three equal parts
    third = len(small_subset) // 3
    lens = [third, third, len(small_subset) - 2*third]
    small_train_subset, small_eval_subset, small_test_subset = random_split(
        small_subset, lens
    )
    
    print(f"Small splits -> train: {len(small_train_subset)}, "
          f"eval: {len(small_eval_subset)}, test: {len(small_test_subset)}")
    
    return small_train_subset, small_eval_subset, small_test_subset

# ============================================================================
# DATALOADER UTILITIES
# ============================================================================

def make_dataloaders(small_train_subset, small_eval_subset, small_test_subset, batch_size):
    """
    Create DataLoader objects for the small subset splits.
    
    Args:
        small_train_subset: Small training subset
        small_eval_subset: Small evaluation subset  
        small_test_subset: Small test subset
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_loader, eval_loader, test_loader)
    """
    train_loader = DataLoader(
        small_train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    eval_loader = DataLoader(
        small_eval_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        small_test_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return train_loader, eval_loader, test_loader

def make_full_train_loader(data_dir, train_transform, eval_transform, batch_size, augment=True):
    """
    Create a DataLoader for the combined train+valid datasets.
    
    Args:
        data_dir: Path to dataset directory
        train_transform: Augmented transform pipeline
        eval_transform: Non-augmented transform pipeline
        batch_size (int): Batch size
        augment (bool): Whether to apply augmentation to both train and valid
        
    Returns:
        DataLoader: Combined train+valid data loader
    """
    if augment:
        # Apply augmentation to both train and valid for final training
        train_aug = datasets.ImageFolder(
            os.path.join(data_dir, "train"), 
            transform=train_transform
        )
        valid_aug = datasets.ImageFolder(
            os.path.join(data_dir, "valid"), 
            transform=train_transform
        )
        combined = ConcatDataset([train_aug, valid_aug])
    else:
        # Use evaluation transforms (no augmentation)
        train_eval = datasets.ImageFolder(
            os.path.join(data_dir, "train"), 
            transform=eval_transform
        )
        valid_eval = datasets.ImageFolder(
            os.path.join(data_dir, "valid"), 
            transform=eval_transform
        )
        combined = ConcatDataset([train_eval, valid_eval])
    
    return DataLoader(
        combined, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

def make_full_loaders(data_dir, eval_transform, batch_size=256):
    """
    Create DataLoaders for full validation and test sets.
    
    Args:
        data_dir: Path to dataset directory
        eval_transform: Evaluation transform pipeline
        batch_size (int): Batch size for evaluation
        
    Returns:
        tuple: (valid_loader_full, test_loader_full)
    """
    valid_ds = datasets.ImageFolder(
        os.path.join(data_dir, "valid"), 
        transform=eval_transform
    )
    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, "test"), 
        transform=eval_transform
    )
    
    valid_loader_full = DataLoader(
        valid_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    test_loader_full = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return valid_loader_full, test_loader_full

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class FeedForwardMLP(nn.Module):
    """
    Feedforward Multilayer Perceptron for CINIC-10 image classification.
    
    This model flattens 32x32x3 images into 3072-dimensional vectors and
    processes them through multiple fully connected layers with ReLU activation
    and dropout regularization.
    
    Architecture:
    - Input: 3072 features (flattened 32x32x3 images)
    - Hidden layers: Configurable sizes with ReLU activation
    - Dropout: Applied after each hidden layer for regularization
    - Output: 10 logits for CINIC-10 classes (softmax applied in loss function)
    
    Args:
        input_size (int): Input feature size (default: 3*32*32 = 3072)
        hidden_sizes (list): List of hidden layer sizes
        num_classes (int): Number of output classes (default: 10)
        dropout (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, input_size=3*32*32, hidden_sizes=[2048, 1024, 512, 256], 
                 num_classes=10, dropout=0.5):
        super(FeedForwardMLP, self).__init__()
        
        layers = []
        in_dim = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        
        # Output layer (no activation - logits for CrossEntropyLoss)
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input: (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = x.view(x.size(0), -1)
        return self.net(x)

# ============================================================================
# REGULARIZATION UTILITIES
# ============================================================================

def l1_penalty(model, device):
    """
    Calculate L1 regularization penalty for model parameters.
    
    L1 regularization adds the sum of absolute values of parameters to the loss,
    encouraging sparsity in the model weights.
    
    Args:
        model (nn.Module): PyTorch model
        device (torch.device): Device for computation
        
    Returns:
        torch.Tensor: L1 penalty value
    """
    l1_norm = torch.tensor(0., device=device)
    for param in model.parameters():
        l1_norm += torch.norm(param, 1)
    return l1_norm

class EarlyStopper:
    """
    Early stopping utility to prevent overfitting.
    
    Monitors a metric (typically validation loss) and stops training when
    the metric stops improving for a specified number of epochs (patience).
    
    Args:
        patience (int): Number of epochs to wait for improvement
        mode (str): 'min' for loss (lower is better), 'max' for accuracy
        min_delta (float): Minimum change to qualify as improvement
    """
    
    def __init__(self, patience=7, mode="min", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
    
    def step(self, value):
        """
        Check if early stopping should be triggered.
        
        Args:
            value (float): Current metric value
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best is None:
            self.best = value
            return False
        
        # Calculate improvement based on mode
        if self.mode == "min":
            improvement = self.best - value  # For loss, lower is better
        else:
            improvement = value - self.best  # For accuracy, higher is better
        
        if improvement > self.min_delta:
            self.best = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, optimizer, criterion, dataloader, device, l1_lambda=0.0):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        dataloader: Training data loader
        device: Device for computation
        l1_lambda (float): L1 regularization strength
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        
        # Add L1 regularization if specified
        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_penalty(model, device)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    
    return avg_loss, accuracy

def evaluate(model, criterion, dataloader, device, class_names=None, 
             log_to_wandb=True, prefix="val"):
    """
    Evaluate the model on a dataset and compute comprehensive metrics.
    
    This function computes loss, accuracy, precision, recall, and F1-score.
    It also generates confusion matrices and logs results to Weights & Biases.
    
    Args:
        model (nn.Module): Trained PyTorch model
        criterion: Loss function
        dataloader: DataLoader for evaluation
        device: Device for computation
        class_names (list): List of class names for confusion matrix
        log_to_wandb (bool): Whether to log metrics to W&B
        prefix (str): Prefix for metric names ("val", "test", etc.)
        
    Returns:
        tuple: (loss, accuracy, precision, recall, f1, y_true, y_pred)
    """
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            logits = model(X)
            loss = criterion(logits, y)
            
            # Accumulate metrics
            running_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Prepare metrics dictionary
    metrics = {
        f"{prefix}_loss": avg_loss,
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1
    }
    
    # Log to Weights & Biases
    if log_to_wandb:
        wandb_log_safe(metrics)
        
        # Generate and log confusion matrix
        if class_names is not None:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
            plt.tight_layout()
            
            wandb_log_safe({f"{prefix}_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)
    
    return avg_loss, accuracy, precision, recall, f1, y_true, y_pred

def run_training(model, train_loader, val_loader, config, device, writer=None):
    """
    Complete training loop with early stopping, learning rate scheduling, and logging.
    
    This function implements a comprehensive training pipeline including:
    - Configurable optimizers (SGD, AdamW)
    - Learning rate scheduling (StepLR, ReduceLROnPlateau)
    - Early stopping based on validation loss
    - Gradient clipping
    - TensorBoard and W&B logging
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config (dict): Training configuration parameters
        device: Device for computation
        writer: TensorBoard SummaryWriter (optional)
        
    Returns:
        tuple: (trained_model, training_history)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer
    if config.get("optimizer", "SGD") == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    else:  # Default to SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"]
        )
    
    # Configure learning rate scheduler
    scheduler = None
    if config.get("scheduler") == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.5)
        )
    elif config.get("scheduler") == "Plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    # Training history and early stopping
    history = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": [], 
        "lr": []
    }
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = config.get("patience", 8)
    
    print(f"Starting training for {config['epochs']} epochs...")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
    print(f"Early stopping patience: {patience}")
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add L1 regularization if specified
            if config.get("l1_lambda", 0.0) > 0.0:
                loss = loss + config["l1_lambda"] * l1_penalty(model, device)
            
            loss.backward()
            
            # Gradient clipping if specified
            if config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            
            optimizer.step()
            
            # Accumulate training metrics
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        train_loss /= total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Logging
        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        wandb_log_safe({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr,
        })
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored model with best validation loss: {best_val_loss:.4f}")
    
    return model, history

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def grid_search(small_train_subset, small_eval_subset, device, out_dir, 
                param_grid=None, grid_config=None):
    """
    Perform grid search for hyperparameter optimization.
    
    This function systematically evaluates different combinations of learning rate,
    weight decay, and batch size on a small subset of data for quick iteration.
    
    Args:
        small_train_subset: Small training subset for quick experiments
        small_eval_subset: Small evaluation subset
        device: Device for computation
        out_dir: Output directory for saving results
        param_grid (dict): Parameter grid for search
        grid_config (dict): Configuration for grid search training
        
    Returns:
        dict: Best hyperparameters found
    """
    if param_grid is None:
        param_grid = {
            "lr": [0.1, 0.01, 0.001],
            "weight_decay": [5e-4, 1e-4, 0.0],
            "batch_size": [64, 128]
        }
    
    if grid_config is None:
        grid_config = {
            "epochs": 12, 
            "momentum": 0.9, 
            "l1_lambda": 0.0, 
            "patience": 4
        }
    
    grid_results = []
    total_combinations = len(param_grid["lr"]) * len(param_grid["weight_decay"]) * len(param_grid["batch_size"])
    
    print(f"Starting grid search with {total_combinations} combinations...")
    print(f"Parameter grid: {param_grid}")
    
    combination_count = 0
    for lr in param_grid["lr"]:
        for wd in param_grid["weight_decay"]:
            for bs in param_grid["batch_size"]:
                combination_count += 1
                print(f"\nGrid search {combination_count}/{total_combinations}: "
                      f"lr={lr}, wd={wd}, bs={bs}")
                
                # Create configuration for this combination
                config = {
                    "lr": lr,
                    "weight_decay": wd,
                    "momentum": grid_config["momentum"],
                    "epochs": grid_config["epochs"],
                    "l1_lambda": grid_config["l1_lambda"],
                    "patience": grid_config["patience"]
                }
                
                # Create data loaders for this batch size
                train_loader = DataLoader(
                    small_train_subset, batch_size=bs, shuffle=True, 
                    num_workers=2, pin_memory=True
                )
                eval_loader = DataLoader(
                    small_eval_subset, batch_size=bs, shuffle=False,
                    num_workers=2, pin_memory=True
                )
                
                # Create and train model
                model = FeedForwardMLP(
                    hidden_sizes=[2048, 1024, 512, 256], 
                    dropout=0.5
                )
                
                # Set up TensorBoard logging
                run_name = f"grid_lr{lr}_wd{wd}_bs{bs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                writer = SummaryWriter(log_dir=os.path.join(out_dir, "runs", run_name))
                
                try:
                    model, history = run_training(
                        model, train_loader, eval_loader, config, device, writer
                    )
                    
                    # Get best validation accuracy
                    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
                    
                    grid_results.append({
                        "params": {"lr": lr, "weight_decay": wd, "batch_size": bs},
                        "val_acc": best_val_acc,
                        "history": history
                    })
                    
                    print(f"Result: Best val_acc = {best_val_acc:.4f}")
                    
                except Exception as e:
                    print(f"Error in grid search combination: {e}")
                    grid_results.append({
                        "params": {"lr": lr, "weight_decay": wd, "batch_size": bs},
                        "val_acc": 0.0,
                        "error": str(e)
                    })
                
                finally:
                    writer.close()
    
    # Find best parameters
    valid_results = [r for r in grid_results if "error" not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x["val_acc"])
        best_params = best_result["params"]
        
        print(f"\nGrid search complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best validation accuracy: {best_result['val_acc']:.4f}")
        
        # Save best hyperparameters
        with open(os.path.join(out_dir, "best_hyperparams.json"), "w") as f:
            json.dump(best_params, f, indent=2)
        
        return best_params
    else:
        print("Grid search failed - no valid results!")
        return {"lr": 0.01, "weight_decay": 0.0, "batch_size": 64}  # Default fallback

# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

def plot_training_history(history, title_prefix="Training History", save_path=None):
    """
    Plot training curves for loss, accuracy, and learning rate.
    
    Args:
        history (dict): Training history dictionary
        title_prefix (str): Title prefix for the plot
        save_path (str): Path to save the plot (optional)
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker='s', markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker='o', markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker='s', markersize=3)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history["lr"], label="Learning Rate", marker='d', markersize=3, color='red')
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_classification_report(y_true, y_pred, class_names, title="Classification Report"):
    """
    Print a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: List of class names
        title (str): Report title
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        digits=4
    ))

def save_training_summary(best_params, small_test_metrics, final_test_metrics, out_dir):
    """
    Save a comprehensive training summary to JSON.
    
    Args:
        best_params (dict): Best hyperparameters found
        small_test_metrics (dict): Metrics on small test set
        final_test_metrics (dict): Metrics on final test set
        out_dir: Output directory
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_hyperparams": best_params,
        "small_test_metrics": small_test_metrics,
        "final_test_metrics": final_test_metrics,
        "model_architecture": {
            "type": "FeedForwardMLP",
            "hidden_sizes": [2048, 1024, 512, 256],
            "dropout": 0.5,
            "input_size": 3072,
            "num_classes": 10
        }
    }
    
    summary_path = os.path.join(out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    return summary_path

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for CINIC-10 MLP training.
    
    This function orchestrates the complete machine learning pipeline:
    1. Environment setup and data exploration
    2. Data preprocessing and subset creation
    3. Hyperparameter optimization via grid search
    4. Training on small subset with best parameters
    5. Final training on combined train+valid data
    6. Comprehensive evaluation and reporting
    """
    print("=" * 80)
    print("CINIC-10 FEEDFORWARD MLP TRAINING PIPELINE")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Setup environment and paths
    device, data_dir, out_dir, wandb_project = setup_environment()
    
    # Explore dataset structure
    print("\n" + "=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    dataset_info = explore_dataset(data_dir)
    print("Dataset exploration summary:")
    print(json.dumps(dataset_info, indent=2))
    
    # Setup data transforms and load datasets
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)
    train_transform, eval_transform, MEAN, STD = setup_data_transforms()
    train_ds, valid_ds, test_ds, class_names = load_datasets(
        data_dir, train_transform, eval_transform
    )
    
    # Create small subset for rapid experimentation
    small_train_subset, small_eval_subset, small_test_subset = create_small_subset(
        train_ds, subset_fraction=0.30, seed=42
    )
    
    # Create full validation and test loaders
    valid_loader_full, test_loader_full = make_full_loaders(
        data_dir, eval_transform, batch_size=256
    )
    
    # Hyperparameter optimization
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # You can skip grid search and use pre-determined best parameters
    # Comment out the grid search section and uncomment the line below if desired
    # best_params = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 64}
    
    best_params = grid_search(
        small_train_subset, small_eval_subset, device, out_dir
    )
    
    # Training on small subset with best parameters
    print("\n" + "=" * 50)
    print("TRAINING ON SMALL SUBSET")
    print("=" * 50)
    
    # Initialize Weights & Biases for small subset training
    wandb.init(
        project=wandb_project,
        name=f"small_subset_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **best_params,
            "model": "FeedForwardMLP",
            "hidden_sizes": [2048, 1024, 512, 256],
            "dropout": 0.5,
            "epochs": 50,
            "patience": 8,
            "phase": "small_subset"
        }
    )
    
    # Create data loaders for small subset
    small_train_loader, small_eval_loader, small_test_loader = make_dataloaders(
        small_train_subset, small_eval_subset, small_test_subset, 
        best_params["batch_size"]
    )
    
    # Train model on small subset
    small_model = FeedForwardMLP(hidden_sizes=[2048, 1024, 512, 256], dropout=0.5)
    small_config = {
        **best_params,
        "epochs": 50,
        "momentum": 0.9,
        "l1_lambda": 0.0,
        "patience": 8
    }
    
    small_model, small_history = run_training(
        small_model, small_train_loader, small_eval_loader, 
        small_config, device, writer=None
    )
    
    # Evaluate on small test set
    criterion = nn.CrossEntropyLoss()
    small_loss, small_acc, small_prec, small_rec, small_f1, y_true_small, y_pred_small = evaluate(
        small_model, criterion, small_test_loader, device,
        class_names=class_names, prefix="small_test"
    )
    
    print(f"\nSmall subset test results:")
    print(f"Accuracy: {small_acc:.4f}, Precision: {small_prec:.4f}, "
          f"Recall: {small_rec:.4f}, F1: {small_f1:.4f}")
    
    # Save small model
    torch.save(small_model.state_dict(), os.path.join(out_dir, "model_small_best.pth"))
    
    # Update W&B summary and finish run
    if wandb.run is not None:
        wandb.run.summary.update({
            "small_test_loss": small_loss,
            "small_test_accuracy": small_acc,
            "small_test_precision": small_prec,
            "small_test_recall": small_rec,
            "small_test_f1": small_f1,
        })
    wandb.finish()
    
    # Final training on combined train+valid
    print("\n" + "=" * 50)
    print("FINAL TRAINING ON COMBINED DATASET")
    print("=" * 50)
    
    # Initialize W&B for final training
    wandb.init(
        project=wandb_project,
        name=f"final_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **best_params,
            "model": "FeedForwardMLP",
            "hidden_sizes": [2048, 1024, 512, 256],
            "dropout": 0.5,
            "epochs": 50,
            "patience": 8,
            "phase": "final_combined",
            "scheduler": "StepLR"
        }
    )
    
    # Create combined training loader
    final_train_loader = make_full_train_loader(
        data_dir, train_transform, eval_transform, 
        best_params["batch_size"], augment=True
    )
    
    # Train final model
    final_model = FeedForwardMLP(hidden_sizes=[2048, 1024, 512, 256], dropout=0.5)
    final_config = {
        **best_params,
        "epochs": 50,
        "momentum": 0.9,
        "l1_lambda": 0.0,
        "patience": 8,
        "scheduler": "StepLR",
        "step_size": 10,
        "gamma": 0.5
    }
    
    # Set up TensorBoard logging
    tb_name = f"final_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "runs", tb_name))
    
    final_model, final_history = run_training(
        final_model, final_train_loader, valid_loader_full, 
        final_config, device, writer=writer
    )
    writer.close()
    
    # Evaluate on held-out test set
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true_test, y_pred_test = evaluate(
        final_model, criterion, test_loader_full, device,
        class_names=class_names, prefix="test"
    )
    
    print(f"\nFinal test set results:")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, "
          f"Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
    
    # Update W&B summary
    if wandb.run is not None:
        wandb.run.summary.update({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
        })
    
    # Save final model
    torch.save(final_model.state_dict(), os.path.join(out_dir, "final_model_combined.pth"))
    
    # Generate comprehensive reports
    print("\n" + "=" * 50)
    print("GENERATING REPORTS")
    print("=" * 50)
    
    # Plot training history
    plot_training_history(
        final_history, 
        title_prefix="Final Combined Training History",
        save_path=os.path.join(out_dir, "final_training_history.png")
    )
    
    # Print classification report
    print_classification_report(
        y_true_test, y_pred_test, class_names, 
        title="Final Test Set Classification Report"
    )
    
    # Save training summary
    small_test_metrics = {
        "accuracy": small_acc, "precision": small_prec, 
        "recall": small_rec, "f1": small_f1
    }
    final_test_metrics = {
        "accuracy": test_acc, "precision": test_prec, 
        "recall": test_rec, "f1": test_f1
    }
    
    save_training_summary(
        best_params, small_test_metrics, final_test_metrics, out_dir
    )
    
    wandb.finish()
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Artifacts saved to: {out_dir}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir {os.path.join(out_dir, 'runs')}")

if __name__ == "__main__":
    main()
