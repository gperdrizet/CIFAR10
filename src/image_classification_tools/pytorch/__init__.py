'''PyTorch utilities for image classification.'''

from image_classification_tools.pytorch.data import (
    load_dataset,
    prepare_splits,
    create_dataloaders,
    generate_augmented_data
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
    create_cnn,
    train_trial,
    create_objective,
    TrialFailedError
)

__all__ = [
    # Data loading and preprocessing
    'load_dataset',
    'prepare_splits',
    'create_dataloaders',
    'generate_augmented_data',
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
    'create_cnn',
    'train_trial',
    'create_objective',
    'TrialFailedError'
]
