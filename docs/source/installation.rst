Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   pip install image-classification-tools

This installs the package and its core dependencies (PyTorch, torchvision, numpy, matplotlib).

Requirements
------------

* Python ≥ 3.10
* PyTorch ≥ 2.0.0
* torchvision ≥ 0.15.0

Optional dependencies
---------------------

For hyperparameter optimization:

.. code-block:: bash

   pip install image-classification-tools[optuna]

Or install with all optional dependencies:

.. code-block:: bash

   pip install image-classification-tools[all]

Install from source
-------------------

To install the development version:

.. code-block:: bash

   git clone https://github.com/gperdrizet/CIFAR10.git
   cd CIFAR10
   pip install -e .

Verify installation
-------------------

.. code-block:: python

   import image_classification_tools
   from image_classification_tools.pytorch import data, training, evaluation

   print("Installation successful")

GPU support
-----------

The package works with both CPU and GPU. To verify CUDA availability:

.. code-block:: python

   import torch
   
   print(f'CUDA available: {torch.cuda.is_available()}')
   
   if torch.cuda.is_available():
       print(f'CUDA version: {torch.version.cuda}')
       print(f'Device count: {torch.cuda.device_count()}')

Next steps
----------

* :doc:`quickstart` - Get started with a quick example
* :doc:`api/index` - Browse the API reference
* :doc:`api/index` - Browse the API reference
