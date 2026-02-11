Data loading
============

.. automodule:: image_classification_tools.pytorch.data
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.data.load_dataset

.. autofunction:: image_classification_tools.pytorch.data.prepare_splits

.. autofunction:: image_classification_tools.pytorch.data.create_dataloaders

.. autofunction:: image_classification_tools.pytorch.data.generate_augmented_data

Overview
--------

The data module provides a flexible three-step data loading workflow:

1. **Load datasets**: Load individual train or test datasets from PyTorch dataset classes or directories
2. **Prepare splits**: Split data into train/val(/test) with configurable sizes
3. **Create dataloaders**: Create DataLoaders with optional memory preloading strategies

Key features:

* Support for torchvision datasets (CIFAR-10, MNIST, etc.) and custom ImageFolder datasets
* Automatic dataset download if not present (for PyTorch built-in datasets)
* Single transform applied to both training and test data
* Flexible splitting: 2-way (train/val) or 3-way (train/val/test) with integer sizes
* Three memory strategies: lazy loading, CPU preloading, or GPU preloading
* Data augmentation with chunking for large datasets
* Configurable batch sizes and workers

Example usage
-------------

Basic workflow (CIFAR-10 with GPU preloading):

.. code-block:: python

   from pathlib import Path
   import torch
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.data import (
       load_dataset, prepare_splits, create_dataloaders
   )

   # Define transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Step 1: Load datasets
   train_dataset = load_dataset(
       data_source=datasets.CIFAR10,
       transform=transform,
       train=True,
       root='./data/cifar10'  # Will download if not present
   )
   
   test_dataset = load_dataset(
       data_source=datasets.CIFAR10,
       transform=transform,
       train=False,
       root='./data/cifar10'
   )

   # Step 2: Prepare splits (2-way: train/val from train_dataset)
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       val_size=10000  # 10,000 images for validation
   )

   # Step 3: Create dataloaders with GPU preloading
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=True,
       device=device
   )

With data augmentation (lazy loading):

.. code-block:: python

   # Define transform with augmentation for training
   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Define transform without augmentation for evaluation
   eval_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Load training data with augmentation
   train_dataset = load_dataset(
       data_source=datasets.CIFAR10,
       transform=train_transform,
       train=True,
       root='./data/cifar10'
   )
   
   # Load test data without augmentation
   test_dataset = load_dataset(
       data_source=datasets.CIFAR10,
       transform=eval_transform,
       train=False,
       root='./data/cifar10'
   )

   # Prepare splits
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       val_size=10000
   )

   # Create dataloaders with lazy loading (no preloading)
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=False,  # Lazy loading for augmentation
       num_workers=4,
       pin_memory=True
   )

3-way split (no separate test set):

.. code-block:: python

   # Define transform
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Load only training data (no test set available)
   train_dataset = load_dataset(
       data_source='./my_dataset',  # Path to directory with train/test subdirs
       transform=transform,
       train=True
   )

   # 3-way split: train/val/test all from train_dataset
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=None,  # Will split test from train_dataset
       val_size=5000,  # 5,000 images for validation
       test_size=5000  # 5,000 images for testing
   )
   # Remaining images will be used for training

