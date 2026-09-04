# Image Classification Tools

A lightweight PyTorch toolkit for building and training image classification models.

## Overview

This package provides utilities for common image classification tasks:

- **Data loading**: Unified ``DataPipeline`` for automatic data preparation with intelligent splitting
- **Model training**: Training loops with progress tracking and validation
- **Evaluation**: Accuracy metrics, confusion matrices, and performance analysis
- **Visualization**: Learning curves, probability distributions, and evaluation plots
- **Hyperparameter optimization**: Optuna integration for automated model tuning

## Installation

```bash
pip install image-classification-tools
```

For hyperparameter optimization, install with the `optuna` extra:

```bash
pip install image-classification-tools[optuna]
```

## Quick start

### Basic usage

```python
import torch
from torchvision import datasets, transforms
from image_classification_tools.pytorch import DataPipeline
from image_classification_tools.pytorch.training import train_model
from image_classification_tools.pytorch.evaluation import evaluate_model

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create data pipeline (handles everything in one call)
loaders = DataPipeline(
    data_source=datasets.MNIST,
    data_dir='./data/pytorch/mnist',
    split='train/val/test',
    val_size=10000,
    batch_size=64,
    train_transform=transform,
    eval_transform=transform,
    preload='gpu'
).get_loaders()

# Access loaders via attributes
train_loader = loaders.train
val_loader = loaders.val
test_loader = loaders.test

# Display summary
print(loaders.split_info())
print(loaders.memory_estimate())

# Define model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)

# Evaluate
accuracy, predictions, labels = evaluate_model(model, test_loader)
print(f'Test accuracy: {accuracy:.2f}%')
```

### Data augmentation

```python
from image_classification_tools.pytorch import DataPipeline

# Define augmentation transforms
pil_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])

tensor_augmentations = transforms.Compose([
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

# Base transform
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create pipeline with augmentation (automatically pregenerated)
loaders = DataPipeline(
    data_source=datasets.CIFAR10,
    data_dir='./data/pytorch/cifar10',
    split='train/val/test',
    batch_size=128,
    train_transform=base_transform,
    eval_transform=base_transform,
    preload='gpu',  # Load augmented data to GPU for fast training
    n_augmentations=5,  # 5 augmented copies per image
    augmented_dataset_name='strong_aug_v1',  # Optional: defaults to 'depth_5'
    pil_augmentations=pil_augmentations,
    tensor_augmentations=tensor_augmentations
).get_loaders()

# Augmented data is saved to: ./data/pytorch/augmented_cifar10/strong_aug_v1/
# Subsequent runs with same augmented_dataset_name will load from cache
```

### Hyperparameter optimization

```python
import torch.nn as nn
from image_classification_tools.pytorch.hyperparameter_optimization import (
    create_objective, MockTrial, TrialFailedError
)
import optuna
from torchvision import datasets

# Define your model factory
def create_cnn(trial, num_classes, in_channels):
    '''Model factory that samples architecture from trial.'''
    n_blocks = trial.suggest_int('n_conv_blocks', 1, 3)
    filters = trial.suggest_categorical('initial_filters', [16, 32, 64])
    # Build your model here...
    return model

# Define search space for training hyperparameters
search_space = {
    'batch_size': [32, 64, 128],
    'learning_rate': (1e-4, 1e-2, 'log'),
    'optimizer': ['Adam', 'SGD'],
    'weight_decay': (1e-6, 1e-3, 'log')
}

# Create objective function
objective = create_objective(
    model_factory=create_cnn,
    data_source=datasets.MNIST,
    data_dir='./data',
    train_transform=transform,
    eval_transform=transform,
    n_epochs=20,
    num_classes=10,
    in_channels=1,
    val_size=10000,
    search_space=search_space
)

# Run optimization with multi-GPU support
study = optuna.create_study(direction='maximize')
n_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
study.optimize(objective, n_trials=50, n_jobs=n_workers, catch=(TrialFailedError,))

# Recreate best model
mock_trial = MockTrial(study.best_params)
best_model = create_cnn(mock_trial, num_classes=10, in_channels=1)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- numpy
- matplotlib
- optuna (optional, for hyperparameter optimization — install via `pip install image-classification-tools[optuna]`)

## Documentation

Full documentation is available at: https://gperdrizet.github.io/CIFAR10/

## Demo project

See a complete example of using this package for CIFAR-10 classification:
https://github.com/gperdrizet/CIFAR10

## License

GPLv3
