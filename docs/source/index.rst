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
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch import DataPipeline, train_model

   # Define transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   # Create data pipeline (handles loading, splitting, and preloading)
   loaders = DataPipeline(
       data_source=datasets.MNIST,
       data_dir='./data/mnist',
       split='train/val/test',
       val_size=10000,
       batch_size=64,
       train_transform=transform,
       eval_transform=transform,
       preload='gpu'
   ).get_loaders()

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
       train_loader=loaders.train,
       val_loader=loaders.val,
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
