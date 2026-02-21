---
license: gpl-3.0
tags:
- image-classification
- pytorch
- cifar10
- computer-vision
- deep-learning
datasets:
- cifar10
metrics:
- accuracy
---

# CIFAR-10 Image Classification Models

This repository contains a collection of PyTorch models trained on the CIFAR-10 dataset, demonstrating progressive improvements in image classification through different architectures and training techniques.

GitHub repository: [gperdrizet/CIFAR10](https://github.com/gperdrizet/CIFAR10)

## Model Overview

| Model | Architecture | Test Accuracy | Parameters |
|-------|-------------|---------------|------------|
| DNN | Fully-connected | ~45% | 1.4M |
| CNN | Basic CNN | ~65% | 122K |
| RGB CNN | CNN with color | ~70% | 370K |
| Optimized CNN | Optuna-tuned architecture | ~75% | Variable |
| Augmented CNN | Optimized + augmentation | ~78% | Variable |
| ResNet50 | Transfer learning | ~85%+ | 23.5M |

*Note: Exact accuracies may vary slightly between training runs*

## Models

### 1. DNN (Deep Neural Network)
**File**: `dnn.pth`

A baseline fully-connected neural network operating on grayscale CIFAR-10 images.

**Architecture**:
- Input: 1024 (32×32 grayscale)
- 4 fully-connected layers: 1024 → 1024 → 256 → 64 → 10
- ReLU activations
- Dropout (0.5) for regularization

**Training**:
- Optimizer: Adam (lr=1e-3)
- Early stopping with patience=15
- Grayscale preprocessing

### 2. CNN (Convolutional Neural Network)
**File**: `cnn.pth`

Basic convolutional architecture to exploit spatial relationships in images.

**Architecture**:
- 3 convolutional blocks with max pooling
- Batch normalization
- Fully-connected classifier

**Training**:
- Optimizer: Adam
- Grayscale preprocessing
- Early stopping

### 3. RGB CNN
**File**: `rgb_cnn.pth`

CNN modified to process full RGB color information.

**Architecture**:
- Similar to CNN but accepts 3-channel RGB input
- Additional filters to handle color information

**Training**:
- RGB preprocessing (no grayscale conversion)
- Same optimization strategy as CNN

### 4. Optimized CNN
**File**: `architecture_optimized_cnn.pth`

Architecture optimized using Optuna hyperparameter search.

**Optimized Parameters**:
- Number of convolutional blocks
- Filters per layer
- Dropout rates
- Kernel sizes

**Training**:
- Optuna study with 100+ trials
- Best architecture selected based on validation accuracy

### 5. Training-Optimized CNN
**File**: `training_optimized_cnn.pth`

Further optimization of training hyperparameters.

**Optimized Parameters**:
- Optimizer type (Adam/SGD/AdamW)
- Learning rate
- Weight decay
- Batch size

### 6. Augmented CNN
**File**: `augmented_cnn.pth`

Combines optimized architecture with data augmentation.

**Data Augmentation**:
- Random horizontal flips
- Random crops with padding
- Color jittering
- Random rotations

**Training**:
- Extended training on augmented dataset
- Best-performing model overall (non-transfer learning)

### 7. ResNet50
**File**: `resnet50.pth`

Transfer learning using pre-trained ResNet50.

**Architecture**:
- ResNet50 backbone pre-trained on ImageNet
- Custom classifier head for CIFAR-10
- Images upscaled to 224×224

**Training**:
- Fine-tuned on CIFAR-10
- Lower learning rate for pre-trained layers
- Achieves highest accuracy

## Dataset

**CIFAR-10** consists of 60,000 32×32 color images in 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

**Split**:
- Training: 40,000 images
- Validation: 10,000 images
- Test: 10,000 images

## Usage

### Loading Models

```python
import torch

# Load a model
model = torch.load('dnn.pth', map_location='cpu')
model.eval()

# Or download from Hugging Face Hub
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id='YOUR_USERNAME/CIFAR10',
    filename='dnn.pth'
)
model = torch.load(model_path, map_location='cpu')
```

### Making Predictions

```python
from torchvision import transforms
from PIL import Image

# Preprocessing for grayscale models (DNN, CNN)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Preprocessing for RGB models
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For ResNet50 (224x224 input)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']
print(f'Predicted: {classes[predicted_class]}')
```

## Training Details

All models were trained with:
- **Framework**: PyTorch 2.0+
- **Loss**: CrossEntropyLoss
- **Validation**: 20% of training data
- **Early Stopping**: Prevents overfitting
- **Hardware**: CUDA-enabled GPU (when available)

### Hyperparameter Optimization

Models 4-6 used Optuna for automated hyperparameter tuning:
- **Search Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 100+ per optimization study
- **Objective**: Validation accuracy
- **Pruning**: MedianPruner for early termination of unpromising trials

## Performance Metrics

Each model includes:
- Test accuracy
- Per-class accuracy
- Confusion matrix
- ROC curves
- Precision-Recall curves
- Class probability distributions

See the companion notebooks for detailed performance analysis.

## Repository Structure

```
CIFAR10/
├── notebooks/          # Jupyter notebooks for each model
├── models/            # Saved model files (.pth)
├── src/               # image-classification-tools package
└── data/              # CIFAR-10 dataset (auto-downloaded)
```

## Installation

Install the companion tools package:

```bash
pip install image-classification-tools
```

Or clone the repository for full notebooks:

```bash
git clone https://github.com/YOUR_USERNAME/CIFAR10.git
cd CIFAR10
```

## Documentation

Full documentation and tutorials: [gperdrizet.github.io/CIFAR10](https://gperdrizet.github.io/CIFAR10)

## Citation

If you use these models or code, please cite:

```bibtex
@software{cifar10_tutorial,
  author = {Your Name},
  title = {CIFAR-10 Image Classification Tutorial},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/CIFAR10}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009
- PyTorch framework
- Optuna hyperparameter optimization library
- HuggingFace Hub for model hosting

## Related Resources

- **Package**: [image-classification-tools on PyPI](https://pypi.org/project/image-classification-tools)
- **Documentation**: [Full Tutorial](https://gperdrizet.github.io/CIFAR10)
- **Repository**: [GitHub](https://github.com/YOUR_USERNAME/CIFAR10)

---

*Last updated: February 2026*
