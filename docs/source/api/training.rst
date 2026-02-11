Training
========

.. automodule:: image_classification_tools.pytorch.training
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.training.train_model

Overview
--------

The training module provides utilities for model training with:

* Progress tracking and reporting
* Training/validation loss and accuracy logging at epoch and batch levels
* Configurable print frequency
* Device specification with lazy loading or pre-loaded data support
* Optional validation (can train without validation data)
* **Optional early stopping** with model checkpoint restoration (**disabled by default**)
* Learning rate scheduler support (cyclic and epoch-based like ReduceLROnPlateau)
* Scheduled LR bounds reduction for cyclic schedulers
* Training history returned as a dictionary

**Important**: Early stopping is disabled by default. Set ``enable_early_stopping=True`` to activate it.

Example usage
-------------

Basic training:

.. code-block:: python

   import torch
   from image_classification_tools.pytorch.training import train_model

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = create_model().to(device)
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       device=device,
       lazy_loading=False,  # Data already on device
       epochs=50,
       print_every=5
   )
   
   # Or enable early stopping
   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       device=device,
       lazy_loading=False,
       enable_early_stopping=True,
       early_stopping_patience=10,
       epochs=50,
       print_every=5
   )

   # Access training history
   print(f"Final train accuracy: {history['train_accuracy'][-1]:.2f}%")
   print(f"Final val accuracy: {history['val_accuracy'][-1]:.2f}%")

With learning rate schedulers and early stopping:

.. code-block:: python

   device = torch.device('cuda')
   model = create_model().to(device)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # Cyclic learning rate (steps per batch)
   cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(
       optimizer, base_lr=0.001, max_lr=0.01, step_size_up=500
   )
   
   # Optional: Schedule LR bounds reduction over time
   lr_schedule = {
       'initial_base_lr': 0.001,
       'initial_max_lr': 0.01,
       'final_base_lr': 0.0001,
       'final_max_lr': 0.001,
       'schedule_epochs': 50  # Reduce bounds over first 50 epochs
   }
   
   # Or use epoch-based scheduler like ReduceLROnPlateau
   epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=5
   )

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       device=device,
       lazy_loading=False,
       cyclic_scheduler=cyclic_scheduler,  # Or use epoch_scheduler
       lr_schedule=lr_schedule,  # Optional bounds reduction
       enable_early_stopping=True,  # Enable early stopping (disabled by default)
       early_stopping_patience=10,
       epochs=100,
       print_every=10
   )
   
   # Access batch-level metrics
   print(f"Total training batches: {len(history['batch_train_loss'])}")
   print(f"Learning rate progression: {history['batch_learning_rates'][:10]}")
