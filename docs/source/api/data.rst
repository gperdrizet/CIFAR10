Data loading
============

.. automodule:: image_classification_tools.pytorch.data
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: image_classification_tools.pytorch.data.DataPipeline
   :members:
   :undoc-members:

.. autoclass:: image_classification_tools.pytorch.data.DataLoaders
   :members:
   :undoc-members:

.. autoclass:: image_classification_tools.pytorch.data.AugmentationStrategy
   :members:
   :undoc-members:

Overview
--------

The data module provides a unified ``DataPipeline`` class that handles all data loading and preparation in a single call. The pipeline automatically detects dataset structure, performs intelligent splitting, and handles augmentation with optimal strategies.

Key features:

* **Auto-detection**: Automatically detects if data source has pre-made train/test splits
* **Outcome-based**: User specifies desired outcome (e.g., ``split='train/val/test'``), pipeline determines the how
* **Intelligent splitting**: Performs minimal operations based on source structure and desired outcome
* **Flexible augmentation**: Three strategies - none, on-the-fly, or pregenerated with parallel processing
* **Memory optimization**: GPU/CPU preloading or lazy loading based on use case
* **Type-safe**: Returns frozen ``DataLoaders`` object with ``.train``, ``.val``, ``.test`` attributes
* **Smart caching**: Reuses pregenerated augmentation with user-provided cache keys
* **Dataset statistics**: Built-in method to compute mean/std for normalization

Example usage
-------------

Basic workflow (CIFAR-10 with GPU preloading):

.. code-block:: python

   from torchvision import datasets, transforms
   from image_classification_tools.pytorch import DataPipeline

   # Define transform
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Create pipeline and get loaders in one call
   loaders = DataPipeline(
       data_source=datasets.CIFAR10,
       split='train/val/test',
       train_transform=transform,
       eval_transform=transform,
       preload='gpu',
       batch_size=128,
       val_size=10000,
       seed=42,
       root='./data/cifar10'  # Will download if not present
   ).get_loaders()

   # Access loaders via attributes
   train_loader = loaders.train
   val_loader = loaders.val
   test_loader = loaders.test

   # Display pipeline summary
   print(loaders.split_info())
   # Output: "Train: 40,000 | Val: 10,000 | Test: 10,000"
   
   print(loaders.memory_estimate())
   # Output: "Estimated memory: ~2.3 GB"

With on-the-fly augmentation (lazy loading):

.. code-block:: python

   from image_classification_tools.pytorch import DataPipeline, AugmentationStrategy

   # Define augmentation transforms
   pil_augmentations = transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.2, contrast=0.2)
   ])
   
   tensor_augmentations = transforms.Compose([
       transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
   ])

   # Define base transforms
   train_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   eval_transform = train_transform  # No augmentation for eval

   # Create pipeline with on-the-fly augmentation
   loaders = DataPipeline(
       data_source=datasets.CIFAR10,
       split='train/val/test',
       train_transform=train_transform,
       eval_transform=eval_transform,
       augmentation=AugmentationStrategy.ON_THE_FLY,
       pil_augmentations=pil_augmentations,
       tensor_augmentations=tensor_augmentations,
       preload=None,  # Must use lazy loading for on-the-fly augmentation
       batch_size=128,
       root='./data/cifar10'
   ).get_loaders()

With pregenerated augmentation (fast training):

.. code-block:: python

   # Create pipeline with pregenerated augmentation
   loaders = DataPipeline(
       data_source=datasets.CIFAR10,
       split='train/val/test',
       train_transform=train_transform,
       eval_transform=eval_transform,
       augmentation=AugmentationStrategy.PREGENERATED,
       pil_augmentations=pil_augmentations,
       tensor_augmentations=tensor_augmentations,
       preload='gpu',  # Can preload since augmentation is fixed
       batch_size=128,
       cache_key='cifar10_standard_aug_v1',  # Reuse cached data if available
       root='./data/cifar10'
   ).get_loaders()
   
   # Second run with same cache_key will skip regeneration
   # and load directly from cache

Computing dataset statistics:

.. code-block:: python

   from image_classification_tools.pytorch import DataPipeline

   # Compute mean and std for normalization
   mean, std = DataPipeline.compute_dataset_stats(
       data_source=datasets.CIFAR10,
       root='./data/cifar10',
       channels=3
   )
   print(f'Mean: {mean}')  # (0.4914, 0.4822, 0.4465)
   print(f'Std: {std}')    # (0.2470, 0.2435, 0.2616)

   # Use computed values in transform
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=mean, std=std)
   ])

Custom datasets (ImageFolder):

.. code-block:: python

   from torchvision.datasets import ImageFolder
   
   # Pipeline auto-detects ImageFolder structure
   loaders = DataPipeline(
       data_source=ImageFolder,
       split='train/val/test',
       train_transform=transform,
       eval_transform=transform,
       preload='cpu',
       batch_size=64,
       val_size=5000,
       root='./my_dataset/train'  # Point to train directory
   ).get_loaders()

