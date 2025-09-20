# CINIC-10 Deep Learning Project

A comprehensive deep learning project implementing both **Multilayer Perceptron (MLP)** and **Convolutional Neural Network (CNN)** architectures for image classification on the CINIC-10 dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline for image classification on the CINIC-10 dataset. It includes two main approaches:

1. **Feedforward MLP**: A baseline multilayer perceptron that flattens images into vectors
2. **CNN Architectures**: Modern convolutional networks optimized for image data

The project demonstrates best practices in deep learning including data preprocessing, hyperparameter optimization, regularization techniques, and comprehensive evaluation.

## 📊 Dataset

**CINIC-10** is an augmented extension of CIFAR-10 that includes additional images from ImageNet.

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: ~270,000 total (90k per split)
- **Splits**: Train (90k), Validation (90k), Test (90k)
- **Resolution**: 32×32 RGB images
- **Source**: [CINIC-10 Dataset](https://www.kaggle.com/datasets/mengcius/cinic10)

### Dataset Statistics
- **Mean**: [0.47889522, 0.47227842, 0.43047404]
- **Std**: [0.24205776, 0.23828046, 0.25874835]

## 📁 Project Structure

```
Prosit1/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Model Design                 # Detailed design document
├── prosit1_mlp.py              # MLP implementation
├── prosit1_cnn.py              # CNN implementation
├── Prosit1.ipynb              # Original Jupyter notebook
├── data/                       # Dataset directory
│   ├── train/                  # Training images
│   ├── valid/                  # Validation images
│   └── test/                   # Test images
├── Output/                     # Training outputs
│   ├── runs/                   # TensorBoard logs
│   ├── *.pth                   # Saved models
│   └── *.json                  # Training summaries
├── DL App/                     # Web application
└── wandb/                      # Weights & Biases logs
```

## 🧠 Models

### 1. Feedforward MLP (`prosit1_mlp.py`)

A fully connected neural network that serves as a baseline:

- **Architecture**: 4 hidden layers [2048, 1024, 512, 256]
- **Input**: Flattened 32×32×3 = 3072 features
- **Activation**: ReLU with dropout (0.5)
- **Regularization**: L1/L2 penalties, early stopping
- **Output**: 10 class logits

### 2. CNN Architectures (`prosit1_cnn.py`)

Multiple CNN implementations optimized for image data:

#### Custom CNN
- **Blocks**: 4 convolutional blocks with batch normalization
- **Features**: Residual connections, global average pooling
- **Regularization**: Dropout, batch normalization, data augmentation

#### ResNet-18
- **Base**: Modified ResNet-18 for 32×32 input
- **Modifications**: Adapted first conv layer and removed max pooling
- **Pretrained**: Optional ImageNet initialization

#### EfficientNet-B0
- **Architecture**: Compound scaling for optimal efficiency
- **Features**: Advanced mobile-optimized design
- **Pretrained**: Optional ImageNet initialization

## ✨ Features

### Data Processing
- **Exploration**: Automated dataset structure analysis
- **Augmentation**: Rotation, flipping, color jitter, random erasing
- **Normalization**: CINIC-10 specific statistics
- **Subset Creation**: 30% subset for rapid experimentation

### Training Pipeline
- **Hyperparameter Optimization**: Grid search with early stopping
- **Advanced Optimizers**: SGD with momentum, AdamW
- **Learning Rate Scheduling**: StepLR, CosineAnnealing, OneCycleLR
- **Regularization**: Dropout, weight decay, label smoothing
- **Mixed Precision**: Automatic mixed precision (AMP) support

### Monitoring & Logging
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking and comparison
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrices**: Detailed per-class analysis

### Evaluation
- **Multiple Phases**: Small subset → Full training → Test evaluation
- **Statistical Analysis**: Classification reports and confusion matrices
- **Model Comparison**: Architecture performance comparison
- **Visualization**: Training curves and result plots

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Prosit1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CINIC-10 dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/mengcius/cinic10)
   - Extract to `./data/` directory
   - Ensure structure: `data/{train,valid,test}/{class_name}/`

## ⚡ Quick Start

### Train MLP Model
```bash
python prosit1_mlp.py
```

### Train CNN Models
```bash
python prosit1_cnn.py
```

### Custom Configuration
```python
# Modify hyperparameters in the script
config = {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 100,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing"
}
```

## 📖 Usage

### Basic Training

```python
from prosit1_mlp import main
from prosit1_cnn import main_cnn

# Train MLP
main()

# Train CNN architectures
main_cnn()
```

### Custom Model Training

```python
from prosit1_cnn import create_cnn_model, run_cnn_training

# Create custom CNN
model = create_cnn_model(
    architecture="custom",
    num_classes=10,
    dropout=0.3,
    use_residual=True
)

# Train with custom config
config = {
    "lr": 0.001,
    "epochs": 50,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing"
}

trained_model, history = run_cnn_training(
    model, train_loader, val_loader, config, device
)
```

### Evaluation Only

```python
from prosit1_mlp import evaluate, FeedForwardMLP
import torch

# Load trained model
model = FeedForwardMLP()
model.load_state_dict(torch.load('Output/final_model_combined.pth'))

# Evaluate
metrics = evaluate(model, criterion, test_loader, device)
```

## 📈 Results

### Expected Performance

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| MLP | ~47% | 36M | ~2 hours |
| Custom CNN | ~75%+ | ~5M | ~3 hours |
| ResNet-18 | ~80%+ | 11M | ~4 hours |
| EfficientNet-B0 | ~82%+ | 5M | ~4 hours |

### Key Findings

1. **CNNs significantly outperform MLPs** on image data due to spatial inductive biases
2. **Pretrained models** provide substantial improvements through transfer learning
3. **Data augmentation** is crucial for generalization on small datasets
4. **Modern architectures** (EfficientNet) achieve better accuracy-efficiency trade-offs

## ⚙️ Configuration

### Environment Variables
```bash
export WANDB_PROJECT="cinic10-experiments"
export CUDA_VISIBLE_DEVICES="0"
```

### Key Hyperparameters

#### MLP Configuration
```python
mlp_config = {
    "hidden_sizes": [2048, 1024, 512, 256],
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 0.0,
    "batch_size": 64,
    "optimizer": "SGD",
    "momentum": 0.9
}
```

#### CNN Configuration
```python
cnn_config = {
    "lr": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing",
    "label_smoothing": 0.1,
    "use_amp": True
}
```

## 🔧 Advanced Features

### Experiment Tracking
- **Weights & Biases**: Automatic experiment logging and comparison
- **TensorBoard**: Real-time training visualization
- **Model Checkpointing**: Automatic saving of best models

### Performance Optimization
- **Mixed Precision Training**: Faster training with AMP
- **Data Loading**: Multi-worker data loading with pin memory
- **Memory Efficiency**: Gradient accumulation for large batch sizes

### Reproducibility
- **Seed Setting**: Consistent random seeds across runs
- **Deterministic Operations**: Reproducible results
- **Environment Logging**: Automatic dependency tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CINIC-10 Dataset**: Darlow et al. for creating the enhanced CIFAR-10 dataset
- **PyTorch Team**: For the excellent deep learning framework
- **Weights & Biases**: For experiment tracking capabilities
- **Research Community**: For the foundational work in computer vision

## 📚 References

1. Darlow, L. N., et al. (2018). CINIC-10 is not ImageNet or CIFAR-10. arXiv preprint arXiv:1810.03505.
2. He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.

---

**Note**: This project is part of a deep learning course and demonstrates comprehensive ML engineering practices. For questions or issues, please open a GitHub issue or contact the maintainers.
