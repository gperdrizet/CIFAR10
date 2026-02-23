'''PyTorch utilities for image classification.'''

from image_classification_tools.pytorch.data import (
    DataPipeline,
    DataLoaders
)
from image_classification_tools.pytorch.evaluation import evaluate_model
from image_classification_tools.pytorch.training import train_model
from image_classification_tools.pytorch.plotting import (
    plot_sample_images,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_class_probability_distributions,
    plot_evaluation_curves,
    plot_optimization_results
)
from image_classification_tools.pytorch.hyperparameter_optimization import (
    train_trial,
    create_objective,
    TrialFailedError,
    MockTrial
)

__all__ = [
    # Data loading and preprocessing
    'DataPipeline',
    'DataLoaders',
    # Model evaluation
    'evaluate_model',
    # Model training
    'train_model',
    # Plotting and visualization
    'plot_sample_images',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_class_probability_distributions',
    'plot_evaluation_curves',
    'plot_optimization_results',
    # Hyperparameter optimization
    'train_trial',
    'create_objective',
    'TrialFailedError',
    'MockTrial'
]
