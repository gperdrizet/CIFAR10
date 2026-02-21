Quick start guide
=================

This guide demonstrates common image classification tasks using the package.

Basic workflow
--------------

The typical workflow is:

1. Prepare your data with ``DataPipeline``
2. Define your model architecture
3. Train the model
4. Evaluate performance

Example: MNIST classification
------------------------------

This example shows the complete workflow using the MNIST dataset.

1. Prepare data
^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch import DataPipeline

   # Define preprocessing
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   # Create data pipeline
   loaders = DataPipeline(
       data_source=datasets.MNIST,
       split='train/val/test',
       train_transform=transform,
       eval_transform=transform,
       preload='gpu',
       batch_size=128,
       val_size=10000,
       download=True,
       root='./data/mnist'
   ).get_loaders()

   # Access loaders
   train_loader = loaders.train
   val_loader = loaders.val
   test_loader = loaders.test
   
   # Display summary
   print(loaders.split_info())

2. Define model
^^^^^^^^^^^^^^^

.. code-block:: python

   import torch.nn as nn

   model = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28 * 28, 512),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(512, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 10)
   )

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

3. Train model
^^^^^^^^^^^^^^

.. code-block:: python

   import torch.optim as optim
   from image_classification_tools.pytorch.training import train_model

   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       device=device,
       epochs=20,
       print_every=5
   )

4. Evaluate model
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from image_classification_tools.pytorch.evaluation import evaluate_model

   test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)
   print(f'Test accuracy: {test_accuracy:.2f}%')

5. Visualize results
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from image_classification_tools.pytorch.plotting import (
       plot_learning_curves, plot_confusion_matrix
   )

   # Learning curves
   fig, axes = plot_learning_curves(history)
   plt.show()

   # Confusion matrix
   class_names = [str(i) for i in range(10)]
   fig, ax = plot_confusion_matrix(true_labels, predictions, class_names)
   plt.show()

Working with custom datasets
-----------------------------

For datasets in ImageFolder format:

.. code-block:: python

   from torchvision.datasets import ImageFolder
   from image_classification_tools.pytorch import DataPipeline

   # Define transform
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
   ])

   # Create pipeline (auto-detects ImageFolder structure)
   loaders = DataPipeline(
       data_source=ImageFolder,
       split='train/val/test',
       train_transform=transform,
       eval_transform=transform,
       preload='cpu',  # Use CPU preload for large datasets
       batch_size=64,
       val_size=5000,
       test_size=5000,
       root='./my_dataset/train'
   ).get_loaders()

Your directory structure should be:

.. code-block:: text

   my_dataset/
   ├── train/
   │   ├── class1/
   │   │   ├── img1.jpg
   │   │   └── img2.jpg
   │   └── class2/
   │       ├── img1.jpg
   │       └── img2.jpg
   └── test/          (optional - will be created via split if not present)
       ├── class1/
       └── class2/

Convolutional neural networks
------------------------------

For image data, CNNs typically perform better than fully connected networks:

.. code-block:: python

   # For 28x28 grayscale images (MNIST)
   model = nn.Sequential(
       nn.Conv2d(1, 32, kernel_size=3, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Conv2d(32, 64, kernel_size=3, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Flatten(),
       nn.Linear(64 * 7 * 7, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   ).to(device)

For color images (3 channels), change the first layer to ``nn.Conv2d(3, 32, ...)``.

Data augmentation
-----------------

Improve generalization with data augmentation. The pipeline supports two strategies:

**On-the-fly augmentation** (different each epoch):

.. code-block:: python

   from image_classification_tools.pytorch import DataPipeline, AugmentationStrategy

   # Define augmentation transforms (applied before base transform)
   pil_augmentations = transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.2, contrast=0.2)
   ])
   
   tensor_augmentations = transforms.Compose([
       transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
   ])
   
   # Base transforms (no augmentation)
   base_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   # Create pipeline with on-the-fly augmentation
   loaders = DataPipeline(
       data_source=datasets.MNIST,
       split='train/val/test',
       train_transform=base_transform,
       eval_transform=base_transform,
       augmentation=AugmentationStrategy.ON_THE_FLY,
       pil_augmentations=pil_augmentations,
       tensor_augmentations=tensor_augmentations,
       preload=None,  # Must use lazy loading for on-the-fly augmentation
       batch_size=128,
       num_workers=4,
       pin_memory=True,
       root='./data/mnist'
   ).get_loaders()

**Pregenerated augmentation** (fixed, faster training):

.. code-block:: python

   # Same augmentation transforms as above
   
   # Create pipeline with pregenerated augmentation
   loaders = DataPipeline(
       data_source=datasets.MNIST,
       split='train/val/test',
       train_transform=base_transform,
       eval_transform=base_transform,
       augmentation=AugmentationStrategy.PREGENERATED,
       pil_augmentations=pil_augmentations,
       tensor_augmentations=tensor_augmentations,
       preload='gpu',  # Can preload since augmentation is fixed
       batch_size=128,
       cache_key='mnist_aug_v1',  # Reuse on subsequent runs
       root='./data/mnist'
   ).get_loaders()
   
   # Second run with same cache_key loads from cache (instant)

Hyperparameter optimization
----------------------------

Use Optuna to find optimal hyperparameters. You define your model factory function
that samples architecture hyperparameters:

.. code-block:: python

   import torch.nn as nn
   import optuna
   from image_classification_tools.pytorch.hyperparameter_optimization import (
       create_objective, MockTrial, TrialFailedError
   )
   from torchvision import datasets

   # Define your model factory
   def create_cnn(trial, num_classes, in_channels):
       '''Model factory that samples architecture from trial.'''
       
       # Sample architecture hyperparameters
       n_blocks = trial.suggest_int('n_conv_blocks', 1, 4)
       filters = trial.suggest_categorical('initial_filters', [16, 32, 64])
       dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
       
       # Build model
       layers = []
       current_channels = in_channels
       
       for i in range(n_blocks):
           out_channels = filters * (2 ** i)
           layers.extend([
               nn.Conv2d(current_channels, out_channels, 3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(dropout)
           ])
           current_channels = out_channels
       
       layers.extend([
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(),
           nn.Linear(current_channels, num_classes)
       ])
       
       return nn.Sequential(*layers)

   # Define search space for training hyperparameters
   search_space = {
       'batch_size': [32, 64, 128, 256],
       'learning_rate': (1e-5, 1e-2, 'log'),
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
       search_space=search_space,
       early_stopping_patience=5
   )

   # Run optimization with multi-GPU support
   study = optuna.create_study(
       direction='maximize',
       storage='sqlite:///optimization.db',
       load_if_exists=True
   )
   
   n_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
   study.optimize(
       objective, 
       n_trials=100, 
       n_jobs=n_workers,
       catch=(TrialFailedError,)
   )

   print(f'Best accuracy: {study.best_value:.2f}%')
   print(f'Best params: {study.best_params}')
   
   # Recreate best model
   mock_trial = MockTrial(study.best_params)
   best_model = create_cnn(mock_trial, num_classes=10, in_channels=1)

Next steps
----------

* See the :doc:`api/index` for detailed function documentation
* Check the `demo project <https://github.com/gperdrizet/CIFAR10>`_ for a complete CIFAR-10 example
