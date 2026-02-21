Hyperparameter optimization
============================

.. automodule:: image_classification_tools.pytorch.hyperparameter_optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The hyperparameter optimization module provides Optuna integration for automated hyperparameter 
tuning with multi-GPU support. It includes:

* **Model factory pattern** - You define the model architecture in your notebook/code
* **Automatic GPU selection** - Distributes trials across available GPUs when n_jobs > 1
* **Flexible search spaces** - Define training hyperparameters via dictionaries
* **Automatic trial pruning** - MedianPruner for faster optimization
* **Error handling** - Graceful handling of OOM errors via ``TrialFailedError``

Key components
--------------

create_objective
~~~~~~~~~~~~~~~~

Factory function that creates an Optuna objective for hyperparameter search with automatic GPU selection:

* Accepts a ``model_factory`` callable that creates models per trial
* Automatically distributes trials across GPUs in round-robin fashion
* Accepts configurable ``data_source``, ``val_size``, ``test_size`` for any dataset
* Creates data loaders per trial with suggested batch size
* Raises ``TrialFailedError`` on dimension collapse or CUDA OOM
* Supports early stopping within trials

train_trial
~~~~~~~~~~~

Trains a model for a single Optuna trial with pruning and early stopping support:

* Reports intermediate values for Optuna's pruner
* Supports optional early stopping on validation loss
* Supports optional learning rate schedulers
* Returns best validation accuracy achieved

TrialFailedError
~~~~~~~~~~~~~~~~

Custom exception raised when a trial fails due to:

* CUDA out-of-memory errors
* Dimension collapse (spatial size becoming 0)
* Other unrecoverable architecture errors

Use with ``study.optimize(catch=(TrialFailedError,))`` to mark trials as **FAIL** 
state instead of crashing the optimization. Failed trials include:

* ``trial.user_attrs['failure_reason']``: Either ``'cuda_oom'`` or ``'dimension_collapse'``
* ``trial.user_attrs['error_message']``: Full error message for debugging

Example usage
-------------

Define your model factory in your notebook/code:

.. code-block:: python

   import torch.nn as nn
   
   def create_cnn(trial, num_classes, in_channels):
       '''Model factory that samples architecture hyperparameters from trial.'''
       
       # Sample architecture hyperparameters
       n_conv_blocks = trial.suggest_int('n_conv_blocks', 3, 10)
       initial_filters = trial.suggest_categorical('initial_filters', [32, 64])
       n_fc_layers = trial.suggest_int('n_fc_layers', 1, 3)
       conv_dropout = trial.suggest_float('conv_dropout_rate', 0.0, 0.4)
       fc_dropout = trial.suggest_float('fc_dropout_rate', 0.2, 0.6)
       
       # Build model (your architecture code here)
       layers = []
       current_channels = in_channels
       
       for i in range(n_conv_blocks):
           out_channels = initial_filters * (2 ** (i // 2))
           layers.extend([
               nn.Conv2d(current_channels, out_channels, 3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(),
               nn.MaxPool2d(2) if (i + 1) % 2 == 0 else nn.Identity(),
               nn.Dropout(conv_dropout)
           ])
           current_channels = out_channels
       
       layers.extend([
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(),
           nn.Linear(current_channels, num_classes)
       ])
       
       return nn.Sequential(*layers)

Create objective and run optimization:

.. code-block:: python

   import optuna
   from torchvision import datasets
   from image_classification_tools.pytorch.hyperparameter_optimization import (
       create_objective, TrialFailedError
   )

   # Define search space for training hyperparameters
   search_space = {
       'batch_size': [64, 128, 256],
       'learning_rate': (1e-4, 1e-2, 'log'),
       'optimizer': ['Adam', 'SGD'],
       'weight_decay': (1e-6, 1e-3, 'log')
   }

   # Create objective
   objective = create_objective(
       model_factory=create_cnn,  # Your model factory function
       data_source=datasets.CIFAR10,
       data_dir='./data',
       train_transform=train_transform,
       eval_transform=eval_transform,
       n_epochs=50,
       num_classes=10,
       in_channels=3,
       val_size=10000,
       search_space=search_space,
       early_stopping_patience=5
   )

   # Create study with storage and pruning
   study = optuna.create_study(
       direction='maximize',
       study_name='my_optimization',
       storage='sqlite:///optimization.db',
       load_if_exists=True,
       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   )

   # Run optimization with multi-GPU support
   n_parallel_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
   
   study.optimize(
       objective, 
       n_trials=200, 
       n_jobs=n_parallel_workers,  # Distributes across GPUs automatically
       catch=(TrialFailedError,)
   )

   print(f"Best accuracy: {study.best_value:.2f}%")
   print("Best params:", study.best_params)

Recreate best model:

.. code-block:: python

   # Create a mock trial to extract best hyperparameters
   class MockTrial:
       def __init__(self, params):
           self.params = params
       def suggest_int(self, name, *args):
           return self.params[name]
       def suggest_categorical(self, name, *args):
           return self.params[name]
       def suggest_float(self, name, *args, **kwargs):
           return self.params[name]

   # Create model with best hyperparameters
   mock_trial = MockTrial(study.best_params)
   best_model = create_cnn(mock_trial, num_classes=10, in_channels=3)

Multi-GPU optimization
----------------------

The ``create_objective`` function automatically distributes trials across available GPUs:

* Each trial gets assigned a GPU via ``device = torch.device(f'cuda:{trial.number % n_gpus}')``
* Set ``n_jobs`` in ``study.optimize()`` to the number of GPUs you want to use
* SQLite storage handles concurrent access (suitable for 2-4 parallel workers)
* For >4 workers, consider PostgreSQL or MySQL for better concurrent write performance

Example:

.. code-block:: python

   # Automatically use all available GPUs
   n_parallel_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
   print(f'Using {n_parallel_workers} parallel workers')
   
   study.optimize(
       objective,
       n_trials=200,
       n_jobs=n_parallel_workers,  # Each worker gets a different GPU
       catch=(TrialFailedError,)
   )

Search space format
-------------------

The search space dictionary supports:

* **List**: Categorical choices - ``[64, 128, 256]``
* **Tuple (2 elements)**: Continuous range - ``(0.0, 0.5)`` for float, ``(1, 8)`` for int
* **Tuple (3 elements)**: Range with scale - ``(1e-5, 1e-1, 'log')`` for log-scaled float

Your model factory can define additional architecture hyperparameters inside the function.

Notes
-----

**Model factory pattern**:

* The model factory receives ``(trial, num_classes, in_channels)``
* It should call ``trial.suggest_*()`` methods to sample hyperparameters
* This gives you complete control over architecture and what to optimize
* Different experiments can have different model factories

**Multi-GPU behavior**:

* GPU selection is automatic via round-robin assignment
* Each parallel worker trains on a different GPU
* No manual ``CUDA_VISIBLE_DEVICES`` configuration needed
* Trials are independent - no inter-GPU communication

**Error handling**:

* Dimension collapse and CUDA OOM errors raise ``TrialFailedError``
* Use ``catch=(TrialFailedError,)`` in ``study.optimize()`` to mark trials as FAIL
* Failure reasons stored as trial attributes: ``trial.user_attrs['failure_reason']``
* Query failed trials: ``[t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]``
