Evaluation
==========

.. automodule:: image_classification_tools.pytorch.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.evaluation.evaluate_model

Overview
--------

The evaluation module provides functions for:

* Model evaluation on test sets
* Prediction generation
* Accuracy calculation
* Automatic device detection from model parameters

Example usage
-------------

Basic evaluation:

.. code-block:: python

   from image_classification_tools.pytorch.evaluation import evaluate_model

   # Evaluate model
   test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)
   
   print(f"Test accuracy: {test_accuracy:.2f}%")

Per-class accuracy:

.. code-block:: python

   # Define your class names
   class_names = ['class1', 'class2', 'class3', ...]

   # Calculate per-class accuracy
   class_correct = {name: 0 for name in class_names}
   class_total = {name: 0 for name in class_names}

   for pred, true in zip(predictions, true_labels):
       class_name = class_names[true]
       class_total[class_name] += 1
       if pred == true:
           class_correct[class_name] += 1

   for name in class_names:
       acc = 100 * class_correct[name] / class_total[name]
       print(f'{name}: {acc:.2f}%')

With visualization:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_confusion_matrix
   import matplotlib.pyplot as plt

   test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)

   fig, ax = plot_confusion_matrix(true_labels, predictions, class_names)
   plt.title(f'Test Accuracy: {test_accuracy:.2f}%')
   plt.show()
