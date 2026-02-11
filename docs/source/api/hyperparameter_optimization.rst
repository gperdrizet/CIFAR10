Hyperparameter optimization
============================

.. automodule:: image_classification_tools.pytorch.hyperparameter_optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The hyperparameter optimization module provides Optuna integration for automated CNN architecture 
search and hyperparameter tuning. It includes:

* **Dynamic CNN architecture** with configurable depth, width, and components
* **Flexible search spaces** defined via dictionaries
* **Automatic trial pruning** for faster optimization
* **Error handling** for OOM and architecture mismatches

Key components
--------------

create_cnn
~~~~~~~~~~

Creates a CNN with dynamic architecture based on hyperparameters:

* Variable number of convolutional blocks (1-10 for 32x32 inputs)
* Filters double every 2 blocks (slower growth for deeper networks)
* Conditional max pooling every 2 blocks (enables deeper architectures)
* Dynamic fully-connected layers with halving sizes
* Separate dropout for conv and FC layers
* Batch normalization after each conv layer
* Adaptive average pooling before classifier

create_objective
~~~~~~~~~~~~~~~~

Factory function that creates an Optuna objective for hyperparameter search:

* Accepts configurable search space dictionary
* Creates data loaders per trial with suggested batch size
* Handles dimension collapse errors gracefully (returns 0.0 score)
* Handles CUDA out-of-memory errors gracefully (returns 0.0 score)
* Supports MedianPruner for early stopping
* Records failure reasons as trial attributes for debugging

train_trial
~~~~~~~~~~~

Trains a model for a single Optuna trial with pruning support.

Example usage
-------------

Basic hyperparameter optimization for MNIST:

.. code-block:: python

   import optuna
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.hyperparameter_optimization import create_objective

   # Define transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   # Define search space
   search_space = {
       'batch_size': [64, 128, 256],
       'n_conv_blocks': (1, 3),  # 1-3 blocks for MNIST, use 1-10 for CIFAR-10
       'initial_filters': [16, 32],
       'n_fc_layers': (1, 3),
       'conv_dropout_rate': (0.1, 0.5),
       'fc_dropout_rate': (0.3, 0.7),
       'learning_rate': (1e-4, 1e-2, 'log'),
       'optimizer': ['Adam', 'SGD'],
       'sgd_momentum': (0.8, 0.99),
       'weight_decay': (1e-6, 1e-3, 'log')
   }

   # Create objective
   objective = create_objective(
       data_dir='./data',
       transform=transform,
       n_epochs=20,
       device=torch.device('cuda'),
       num_classes=10,
       in_channels=1,
       search_space=search_space
   )

   # Run optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=50)

   # Get best parameters
   print(f"Best accuracy: {study.best_trial.value:.2f}%")
   print("Best hyperparameters:", study.best_trial.params)

With persistent storage:

.. code-block:: python

   from pathlib import Path

   # SQLite storage for resumable studies
   storage_path = Path('./optimization.db')
   storage_url = f'sqlite:///{storage_path}'

   study = optuna.create_study(
       study_name='cnn_optimization',
       direction='maximize',
       storage=storage_url,
       load_if_exists=True,  # Resume if interrupted
       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   )

   study.optimize(objective, n_trials=200)

Creating the final model:

.. code-block:: python

   from image_classification_tools.pytorch.hyperparameter_optimization import create_cnn

   best_params = study.best_trial.params

   model = create_cnn(
       n_conv_blocks=best_params['n_conv_blocks'],
       initial_filters=best_params['initial_filters'],
       n_fc_layers=best_params['n_fc_layers'],
       conv_dropout_rate=best_params['conv_dropout_rate'],
       fc_dropout_rate=best_params['fc_dropout_rate'],
       num_classes=10,
       in_channels=3
   ).to(device)

Search space format
-------------------

The search space dictionary supports three formats:

* **List**: Categorical choices - ``[64, 128, 256]``
* **Tuple (2 elements)**: Continuous range - ``(0.0, 0.5)`` for float, ``(1, 8)`` for int
* **Tuple (3 elements)**: Range with scale - ``(1e-5, 1e-1, 'log')`` for log-scaled float

Example search space for CIFAR-10:

* Batch size: [64, 128, 256, 512]
* Conv blocks: 1-10 (pools every 2 blocks)
* Initial filters: [16, 32, 64, 128]
* FC layers: 1-5
* Conv dropout: 0.1-0.5
* FC dropout: 0.3-0.7
* Learning rate: 1e-5 to 1e-2 (log scale)
* Optimizer: ['Adam', 'SGD', 'RMSprop'] (optional)
* SGD momentum: 0.8-0.99 (optional)
* Weight decay: 1e-6 to 1e-3 (log scale, optional)

Notes
-----

**Architectural constraints**:

* For 32x32 images (CIFAR-10), support up to 10 conv blocks due to conditional pooling
* Max pooling occurs every 2 blocks, allowing deeper networks without dimension collapse
* Filters double every 2 blocks instead of every block for parameter efficiency
* Adaptive average pooling handles variable spatial dimensions before classifier

**Error handling**:

* Dimension collapse errors (e.g., spatial size becoming 0) return score of 0.0
* CUDA out-of-memory errors are caught and return score of 0.0
* Failure reasons stored as trial attributes: ``trial.user_attrs['failure_reason']``
