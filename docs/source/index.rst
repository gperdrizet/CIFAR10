Image classification tools documentation
=========================================

**image-classification-tools** is a lightweight PyTorch toolkit for building and training image classification models.

The package provides utilities for:

* Loading and preprocessing image datasets
* Training models with validation tracking
* Evaluating model performance
* Visualizing results and metrics
* Optimizing hyperparameters with Optuna

Who should use this
-------------------

This package is for developers who need to:

* Build image classifiers for custom datasets
* Prototype and compare different model architectures
* Automate hyperparameter tuning
* Evaluate and visualize model performance

The API works with any image classification task, from small datasets like MNIST to larger custom collections.

Installation
------------

.. code-block:: bash

   pip install image-classification-tools

Quick example
-------------

Minimal example classifying MNIST digits:

.. code-block:: python

   import torch
   from pathlib import Path
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.data import (
       load_datasets, prepare_splits, create_dataloaders
   )
   from image_classification_tools.pytorch.training import train_model

   # Load data
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   # Load, split, and create dataloaders
   train_dataset, test_dataset = load_datasets(
       data_source=datasets.MNIST,
       train_transform=transform,
       eval_transform=transform,
       download=True,
       root=Path('./data/mnist')
   )
   
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset, test_dataset, train_val_split=0.8
   )
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=64,
       preload_to_memory=True,
       device=device
   )

   # Define model
   model = torch.nn.Sequential(
       torch.nn.Flatten(),
       torch.nn.Linear(784, 128),
       torch.nn.ReLU(),
       torch.nn.Linear(128, 10)
   )

   # Train
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters())
   
   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       epochs=10
   )

Demo project
------------

For a complete example, see the CIFAR-10 classification demo: https://github.com/gperdrizet/CIFAR10

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project links

   GitHub Repository <https://github.com/gperdrizet/CIFAR10>
   PyPI Package <https://pypi.org/project/image-classification-tools>
   Issue Tracker <https://github.com/gperdrizet/CIFAR10/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
