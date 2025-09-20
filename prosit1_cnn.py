#!/usr/bin/env python3
"""
CINIC-10 Convolutional Neural Network Implementation

This script implements a CNN architecture for CINIC-10 image classification,
providing a more suitable approach for image data compared to the MLP baseline.
The CNN leverages spatial hierarchies and translation invariance to achieve
better performance on image classification tasks.

Key Features:
- Modern CNN architecture with batch normalization
- Residual connections for improved gradient flow
- Data augmentation and regularization techniques
- Comprehensive training pipeline with monitoring
- Transfer learning capabilities
- Model ensemble support

Architecture Options:
1. Custom CNN: Lightweight architecture designed for CINIC-10
2. ResNet-18: Deep residual network with skip connections
3. EfficientNet-B0: Efficient architecture with compound scaling

Date: 2025
"""

import os
import json
import copy
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# Import utilities from MLP script
from prosit1_mlp import (
    setup_environment, set_random_seeds, wandb_log_safe,
    explore_dataset, load_datasets, create_small_subset,
    make_dataloaders, make_full_train_loader, make_full_loaders,
    evaluate, plot_training_history, print_classification_report,
    save_training_summary
)

# ============================================================================
# CNN ARCHITECTURES
# ============================================================================

class CINIC10_CNN(nn.Module):
    """
    Custom CNN architecture optimized for CINIC-10 dataset.
    
    This architecture is specifically designed for 32x32 RGB images with
    10 classes. It uses modern techniques like batch normalization,
    dropout, and global average pooling.
    
    Architecture:
    - 4 Convolutional blocks with increasing channels
    - Batch normalization after each conv layer
    - ReLU activation and dropout for regularization
    - Global average pooling instead of fully connected layers
    - Residual connections in deeper blocks
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        dropout (float): Dropout probability (default: 0.3)
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(self, num_classes=10, dropout=0.3, use_residual=True):
        super(CINIC10_CNN, self).__init__()
        
        self.use_residual = use_residual
        
        # First convolutional block: Extract low-level features
        # Input: 3x32x32 -> Output: 64x16x16
        # Two 3x3 convolutions capture edge and texture information
        # BatchNorm stabilizes training, MaxPool reduces spatial dimensions
        self.conv1 = nn.Sequential(
            # First conv layer: 3 input channels (RGB) -> 64 feature maps
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),  # Normalize activations for stable training
            nn.ReLU(inplace=True),  # Non-linear activation
            
            # Second conv layer: Refine features within same spatial resolution
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample: 32x32 -> 16x16, reduce computational load
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Spatial dropout: Randomly zero out entire feature maps for regularization
            nn.Dropout2d(dropout)
        )
        
        # Second convolutional block: Mid-level feature extraction
        # Input: 64x16x16 -> Output: 128x8x8
        # Increased channel depth captures more complex patterns
        self.conv2 = nn.Sequential(
            # Increase feature depth: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Refine features at current resolution
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Further downsample: 16x16 -> 8x8
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )
        
        # Third convolutional block: High-level feature extraction
        # Input: 128x8x8 -> Output: 256x4x4
        # Captures object parts and semantic information
        self.conv3 = nn.Sequential(
            # Double feature depth: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Learn complex feature combinations
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Downsample: 8x8 -> 4x4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )
        
        # Fourth convolutional block: Abstract feature extraction
        # Input: 256x4x4 -> Output: 512x1x1
        # Captures high-level semantic concepts for classification
        self.conv4 = nn.Sequential(
            # Maximum feature depth: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Final feature refinement
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling: 4x4 -> 1x1
            # Reduces overfitting compared to fully connected layers
            # Each of 512 channels becomes a single value
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Residual connection layers for gradient flow improvement
        # 1x1 convolutions match dimensions between blocks for skip connections
        # Helps with vanishing gradient problem in deeper networks
        if use_residual:
            # Match dimensions: 64 -> 128 channels, downsample 2x
            self.residual1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
            # Match dimensions: 128 -> 256 channels, downsample 2x
            self.residual2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
            # Match dimensions: 256 -> 512 channels, downsample 2x
            self.residual3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        
        # Classification head: Convert features to class predictions
        # Input: 512 features -> Output: num_classes logits
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # Final regularization before classification
            nn.Linear(512, num_classes)  # Linear transformation to class scores
        )
        
        # Initialize network weights using best practices
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the CNN network with detailed feature extraction.
        
        The network processes images through four convolutional blocks with
        progressively increasing feature depth and decreasing spatial resolution.
        Optional residual connections help with gradient flow in deeper layers.
        
        Feature extraction hierarchy:
        1. Low-level: Edges, textures, simple patterns
        2. Mid-level: Object parts, shapes, local structures  
        3. High-level: Object components, semantic patterns
        4. Abstract: Class-specific features for classification
        
        Args:
            x (torch.Tensor): Input batch of images, shape (batch_size, 3, 32, 32)
                             Values should be normalized to [-1, 1] or [0, 1] range
            
        Returns:
            torch.Tensor: Raw class logits, shape (batch_size, num_classes)
                         Apply softmax for probabilities: F.softmax(logits, dim=1)
        """
        # First convolutional block: Extract low-level features
        # Input: (B, 3, 32, 32) -> Output: (B, 64, 16, 16)
        # Captures edges, corners, basic textures
        out1 = self.conv1(x)
        
        # Second convolutional block: Mid-level feature extraction
        # Input: (B, 64, 16, 16) -> Output: (B, 128, 8, 8)
        # Combines low-level features into more complex patterns
        out2 = self.conv2(out1)
        
        # Optional residual connection: Helps gradient flow and feature reuse
        # Allows network to learn identity mapping if needed
        if self.use_residual:
            # Project out1 to match out2 dimensions: (B, 64, 16, 16) -> (B, 128, 8, 8)
            residual1 = self.residual1(out1)
            # Element-wise addition combines direct and processed features
            out2 = out2 + residual1
        
        # Third convolutional block: High-level feature extraction  
        # Input: (B, 128, 8, 8) -> Output: (B, 256, 4, 4)
        # Captures object parts and semantic structures
        out3 = self.conv3(out2)
        
        # Second residual connection for deeper gradient flow
        if self.use_residual:
            # Project out2 to match out3 dimensions: (B, 128, 8, 8) -> (B, 256, 4, 4)
            residual2 = self.residual2(out2)
            out3 = out3 + residual2
        
        # Fourth convolutional block: Abstract feature extraction
        # Input: (B, 256, 4, 4) -> Output: (B, 512, 1, 1)
        # Global average pooling creates class-specific feature representations
        out4 = self.conv4(out3)
        
        # Final residual connection with adaptive pooling
        if self.use_residual:
            # Project and pool out3 to match out4: (B, 256, 4, 4) -> (B, 512, 1, 1)
            residual3 = F.adaptive_avg_pool2d(self.residual3(out3), (1, 1))
            out4 = out4 + residual3
        
        # Flatten feature maps for classification
        # Convert (B, 512, 1, 1) -> (B, 512)
        # Each sample now represented by 512-dimensional feature vector
        out = out4.view(out4.size(0), -1)
        
        # Classification layer: Map features to class scores
        # Input: (B, 512) -> Output: (B, num_classes)
        # Higher scores indicate higher confidence for that class
        out = self.classifier(out)
        
        return out

class ResNet18_CINIC10(nn.Module):
    """
    ResNet-18 adapted for CINIC-10 dataset.
    
    This uses the standard ResNet-18 architecture but modifies the first
    layer and final classifier for the CINIC-10 dataset characteristics.
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        pretrained (bool): Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet18_CINIC10, self).__init__()
        
        # Load ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for 32x32 input (instead of 224x224)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                                       padding=1, bias=False)
        
        # Remove the first max pooling layer (not needed for 32x32)
        self.backbone.maxpool = nn.Identity()
        
        # Modify classifier for CINIC-10
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass through ResNet-18."""
        return self.backbone(x)

class EfficientNet_CINIC10(nn.Module):
    """
    EfficientNet-B0 adapted for CINIC-10 dataset.
    
    EfficientNet uses compound scaling to balance network depth, width,
    and resolution for optimal efficiency.
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        pretrained (bool): Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNet_CINIC10, self).__init__()
        
        try:
            # Try to load EfficientNet (requires timm or torchvision >= 0.11)
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            
            # Modify classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        except ImportError:
            print("EfficientNet not available, falling back to ResNet-18")
            # Fallback to ResNet-18 if EfficientNet is not available
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                                           padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass through EfficientNet."""
        return self.backbone(x)

# ============================================================================
# CNN-SPECIFIC DATA AUGMENTATION
# ============================================================================

def setup_cnn_transforms():
    """
    Create enhanced data transformation pipelines optimized for CNN training.
    
    CNN-specific augmentations include more aggressive spatial transformations
    and advanced techniques like Cutout and MixUp (if implemented).
    
    Returns:
        tuple: (train_transform, eval_transform, mean, std)
    """
    # CINIC-10 dataset statistics
    MEAN = [0.47889522, 0.47227842, 0.43047404]
    STD = [0.24205776, 0.23828046, 0.25874835]
    
    # Enhanced training transforms for CNN
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                             saturation=0.2, hue=0.02),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), 
                               ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    # Standard evaluation transforms
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    return train_transform, eval_transform, MEAN, STD

# ============================================================================
# CNN TRAINING UTILITIES
# ============================================================================

def run_cnn_training(model, train_loader, val_loader, config, device, writer=None):
    """
    CNN-specific training loop with advanced techniques.
    
    This function extends the basic training loop with CNN-specific
    optimizations like cosine annealing, warmup, and mixed precision training.
    
    Args:
        model (nn.Module): CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config (dict): Training configuration
        device: Device for computation
        writer: TensorBoard writer (optional)
        
    Returns:
        tuple: (trained_model, training_history)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    
    # Configure optimizer with different learning rates for different parts
    if config.get("optimizer", "AdamW") == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:  # SGD with momentum
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"],
            nesterov=True
        )
    
    # Advanced learning rate scheduling
    if config.get("scheduler") == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
        )
    elif config.get("scheduler") == "OneCycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["lr"], 
            steps_per_epoch=len(train_loader), 
            epochs=config["epochs"]
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("step_size", 30), 
            gamma=config.get("gamma", 0.1)
        )
    
    # Training history
    history = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": [], 
        "lr": []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = config.get("patience", 10)
    
    print(f"Starting CNN training for {config['epochs']} epochs...")
    print(f"Model: {type(model).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__}")
    
    # Mixed precision training (if available)
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Using Automatic Mixed Precision (AMP)")
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                
                if config.get("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if config.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                
                optimizer.step()
            
            # Update learning rate for OneCycle
            if config.get("scheduler") == "OneCycle":
                scheduler.step()
            
            # Accumulate metrics
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        train_loss /= total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Learning rate scheduling (except OneCycle which is per-batch)
        if config.get("scheduler") != "OneCycle":
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
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
        print(f"Restored model with best validation accuracy: {best_val_acc:.4f}")
    
    return model, history

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_cnn_model(architecture="custom", num_classes=10, **kwargs):
    """
    Factory function to create different CNN architectures.
    
    Args:
        architecture (str): Architecture type ("custom", "resnet18", "efficientnet")
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for model creation
        
    Returns:
        nn.Module: CNN model
    """
    if architecture.lower() == "custom":
        return CINIC10_CNN(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.3),
            use_residual=kwargs.get("use_residual", True)
        )
    elif architecture.lower() == "resnet18":
        return ResNet18_CINIC10(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True)
        )
    elif architecture.lower() == "efficientnet":
        return EfficientNet_CINIC10(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

# ============================================================================
# MAIN CNN TRAINING PIPELINE
# ============================================================================

def main_cnn():
    """
    Main execution pipeline for CINIC-10 CNN training.
    
    This function demonstrates how to train different CNN architectures
    on the CINIC-10 dataset with comprehensive evaluation.
    """
    print("=" * 80)
    print("CINIC-10 CNN TRAINING PIPELINE")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Setup environment
    device, data_dir, out_dir, wandb_project = setup_environment()
    
    # CNN-specific transforms
    train_transform, eval_transform, MEAN, STD = setup_cnn_transforms()
    
    # Load datasets
    train_ds, valid_ds, test_ds, class_names = load_datasets(
        data_dir, train_transform, eval_transform
    )
    
    # Create data loaders
    batch_size = 128  # Larger batch size for CNN
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Training configurations for different architectures
    architectures = {
        "custom": {
            "model_kwargs": {"dropout": 0.3, "use_residual": True},
            "config": {
                "lr": 0.001,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealing",
                "epochs": 100,
                "patience": 15,
                "label_smoothing": 0.1,
                "use_amp": True,
                "grad_clip": 1.0
            }
        },
        "resnet18": {
            "model_kwargs": {"pretrained": True},
            "config": {
                "lr": 0.01,
                "weight_decay": 1e-4,
                "optimizer": "SGD",
                "momentum": 0.9,
                "scheduler": "StepLR",
                "step_size": 30,
                "gamma": 0.1,
                "epochs": 100,
                "patience": 15,
                "use_amp": True,
                "grad_clip": 1.0
            }
        }
    }
    
    results = {}
    
    # Train each architecture
    for arch_name, arch_config in architectures.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {arch_name.upper()} ARCHITECTURE")
        print(f"{'='*60}")
        
        # Initialize W&B
        wandb.init(
            project=wandb_project,
            name=f"cnn_{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": arch_name,
                "batch_size": batch_size,
                **arch_config["config"],
                **arch_config["model_kwargs"]
            }
        )
        
        # Create model
        model = create_cnn_model(
            architecture=arch_name,
            num_classes=len(class_names),
            **arch_config["model_kwargs"]
        )
        
        print(f"Model: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Set up TensorBoard logging
        tb_name = f"cnn_{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "runs", tb_name))
        
        # Train model
        model, history = run_cnn_training(
            model, train_loader, valid_loader, 
            arch_config["config"], device, writer
        )
        writer.close()
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(
            model, criterion, test_loader, device,
            class_names=class_names, prefix="test"
        )
        
        print(f"\n{arch_name.upper()} Test Results:")
        print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}")
        print(f"Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
        
        # Save model and results
        model_path = os.path.join(out_dir, f"cnn_{arch_name}_best.pth")
        torch.save(model.state_dict(), model_path)
        
        # Plot training history
        plot_training_history(
            history,
            title_prefix=f"{arch_name.upper()} Training History",
            save_path=os.path.join(out_dir, f"cnn_{arch_name}_history.png")
        )
        
        # Store results
        results[arch_name] = {
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "model_path": model_path,
            "parameters": sum(p.numel() for p in model.parameters())
        }
        
        # Update W&B summary
        if wandb.run is not None:
            wandb.run.summary.update({
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,
                "model_parameters": results[arch_name]["parameters"]
            })
        
        wandb.finish()
    
    # Compare results
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    
    comparison_df = []
    for arch, result in results.items():
        comparison_df.append({
            "Architecture": arch,
            "Test Accuracy": f"{result['test_accuracy']:.4f}",
            "Test F1": f"{result['test_f1']:.4f}",
            "Parameters": f"{result['parameters']:,}"
        })
    
    # Print comparison table
    print(f"{'Architecture':<15} {'Test Accuracy':<15} {'Test F1':<10} {'Parameters':<15}")
    print("-" * 60)
    for row in comparison_df:
        print(f"{row['Architecture']:<15} {row['Test Accuracy']:<15} "
              f"{row['Test F1']:<10} {row['Parameters']:<15}")
    
    # Save comparison results
    comparison_path = os.path.join(out_dir, "cnn_architecture_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {out_dir}")
    print(f"Comparison saved to: {comparison_path}")

if __name__ == "__main__":
    main_cnn()
