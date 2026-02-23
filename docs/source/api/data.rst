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

Overview
--------

The data module provides a unified ``DataPipeline`` class that handles all data loading and preparation in a single call. The pipeline automatically detects dataset structure, performs intelligent splitting, and handles augmentation with pregeneration and caching.

Key features:

* **Auto-detection**: Automatically detects if data source has pre-made train/test splits
* **Outcome-based**: User specifies desired outcome (e.g., ``split='train/val/test'``), pipeline determines the how
* **Intelligent splitting**: Performs minimal operations based on source structure and desired outcome
* **Pregenerated augmentation**: Augmented data is generated once and saved to disk for reuse
* **Memory optimization**: GPU/CPU preloading or lazy loading based on use case
* **Type-safe**: Returns frozen ``DataLoaders`` object with ``.train``, ``.val``, ``.test`` attributes
* **Smart caching**: Reuses pregenerated augmentation across training runs
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
       data_dir='./data/pytorch/cifar10',  # Will download if not present
       split='train/val/test',
       val_size=10000,
       batch_size=128,
       train_transform=transform,
       eval_transform=transform,
       preload='gpu',
       seed=42
   ).get_loaders()

   # Access loaders via attributes
   train_loader = loaders.train
   val_loader = loaders.val
   test_loader = loaders.test

   # Display pipeline summary
   print(loaders.total_samples())
   # Output: {'train': 40000, 'val': 10000, 'test': 10000}
   
   print(loaders.memory_estimate())
   # Output: 2.3 (GB)

With data augmentation (pregenerated):

.. code-block:: python

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

   # Define base transforms
   base_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Create pipeline with augmentation
   # Augmented data is automatically pregenerated and saved to disk
   loaders = DataPipeline(
       data_source=datasets.CIFAR10,
       data_dir='./data/pytorch/cifar10',
       split='train/val/test',
       batch_size=128,
       train_transform=base_transform,
       eval_transform=base_transform,
       preload='gpu',  # Preload augmented data to GPU for fast training
       n_augmentations=5,  # Create 5 augmented copies per image
       augmented_dataset_name='strong_aug_v1',  # Optional: defaults to 'depth_5'
       pil_augmentations=pil_augmentations,
       tensor_augmentations=tensor_augmentations
   ).get_loaders()
   
   # Augmented data saved to: ./data/pytorch/augmented_cifar10/strong_aug_v1/
   # Subsequent runs with same augmented_dataset_name load from cache
   # Use force_regenerate=True to regenerate cached data

Computing dataset statistics:

.. code-block:: python

   from image_classification_tools.pytorch import DataPipeline

   # Compute mean and std for normalization
   mean, std = DataPipeline.compute_dataset_stats(
       data_source=datasets.CIFAR10,
       data_dir='./data/pytorch/cifar10',
       num_samples=5000
   )
   print(f'Mean: {mean}')  # (0.4914, 0.4822, 0.4465)
   print(f'Std: {std}')    # (0.2470, 0.2435, 0.2616)

   # Use computed values in transform
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=mean, std=std)
   ])

Custom datasets (directory-based):

.. code-block:: python

   # Pipeline auto-detects directory structure
   loaders = DataPipeline(
       data_source='./my_dataset',  # Path to dataset directory
       data_dir='./my_dataset',
       split='train/val/test',
       val_size=5000,
       batch_size=64,
       train_transform=transform,
       eval_transform=transform,
       preload='cpu'
   ).get_loaders()
