'''Plotting functions for image classification models.

This module provides visualization utilities for analyzing image classification models,
including sample images, learning curves, confusion matrices, and evaluation metrics.
'''

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


def plot_sample_images(
    dataset: Dataset,
    class_names: list[str],
    nrows: int = 2,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot sample images from a dataset.
    
    Automatically handles both grayscale (1 channel) and RGB (3 channel) images.
    
    Args:
        dataset: PyTorch dataset containing (image, label) tuples.
        class_names: List of class names for labeling.
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        figsize: Figure size (width, height). Defaults to (ncols*1.5, nrows*1.5).
        
    Returns:
        Tuple of (figure, axes array).
    '''
    if figsize is None:
        figsize = (ncols * 1.5, nrows * 1.5)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Get image and label from dataset
        img, label = dataset[i]
        
        # Unnormalize for plotting
        img = img * 0.5 + 0.5
        img = img.numpy()
        
        # Handle grayscale vs RGB images
        if img.shape[0] == 1:

            # Grayscale: squeeze channel dimension
            img = img.squeeze()
            ax.imshow(img, cmap='gray')
    
        else:

            # RGB: transpose from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
        
        ax.set_title(class_names[label])
        ax.axis('off')

    plt.tight_layout()
    
    return fig, axes


def plot_learning_curves(
    history: dict[str, list[float]],
    figsize: tuple[float, float] = (8, 3)
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 
                 'train_accuracy', and 'val_accuracy' lists.
        figsize: Figure size (width, height).
        
    Returns:
        Tuple of (figure, axes array).
    '''
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Find epoch with best validation loss
    best_epoch = np.argmin(history['val_loss'])

    axes[0].set_title('Loss')
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Model')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (cross-entropy)')
    axes[0].legend(loc='best')

    axes[1].set_title('Accuracy')
    axes[1].plot(history['train_accuracy'], label='Train')
    axes[1].plot(history['val_accuracy'], label='Validation')
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Model')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='best')

    plt.tight_layout()
    
    return fig, axes


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    figsize: tuple[float, float] = (8, 8),
    cmap: str = 'Blues'
) -> tuple[plt.Figure, plt.Axes]:
    '''Plot a confusion matrix heatmap.
    
    Args:
        true_labels: Array of true class labels.
        predictions: Array of predicted class labels.
        class_names: List of class names for labeling.
        figsize: Figure size (width, height).
        cmap: Colormap for the heatmap.
        
    Returns:
        Tuple of (figure, axes).
    '''
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, predictions)
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('Confusion matrix')
    im = ax.imshow(cm, cmap=cmap)

    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.tight_layout()
    
    return fig, ax


def plot_class_probability_distributions(
    all_probs: np.ndarray,
    class_names: list[str],
    nrows: int = 2,
    ncols: int = 5,
    figsize: tuple[float, float] = (8, 3),
    bins: int = 50,
    color: str = 'black'
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot predicted probability distributions for each class.
    
    Args:
        all_probs: Array of shape (n_samples, n_classes) with predicted probabilities.
        class_names: List of class names for labeling.
        nrows: Number of rows in the subplot grid.
        ncols: Number of columns in the subplot grid.
        figsize: Figure size (width, height).
        bins: Number of histogram bins.
        color: Histogram bar color.
        
    Returns:
        Tuple of (figure, axes array).
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    fig.suptitle('Predicted probability distributions by class')
    fig.supxlabel('Predicted probability')
    fig.supylabel('Count')

    axes = axes.flatten()

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):

        # Get probabilities for this class across all samples
        class_probs = all_probs[:, i]
        
        # Plot histogram
        ax.hist(class_probs, bins=bins, color=color)
        ax.set_title(class_name)
        ax.set_xlim(0, 1)
        ax.set_yscale('log')

    plt.tight_layout()
    
    return fig, axes


def plot_evaluation_curves(
    true_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: list[str],
    figsize: tuple[float, float] = (8, 4)
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    '''Plot ROC and Precision-Recall curves for multi-class classification.
    
    Args:
        true_labels: Array of true class labels.
        all_probs: Array of shape (n_samples, n_classes) with predicted probabilities.
        class_names: List of class names for labeling.
        figsize: Figure size (width, height).
        
    Returns:
        Tuple of (figure, (ax1, ax2)).
    '''
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    # Binarize true labels for one-vs-rest evaluation
    y_test_bin = label_binarize(true_labels, classes=range(len(class_names)))

    # Create figure with ROC and PR curves side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ROC curves for each class
    ax1.set_title('ROC curves (one-vs-rest)')

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        _ = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=class_name)

    ax1.plot([0, 1], [0, 1], 'k--', label='random classifier')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # Plot Precision-Recall curves for each class
    ax2.set_title('Precision-recall curves (one-vs-rest)')

    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_test_bin[:, i], all_probs[:, i])
        ax2.plot(recall, precision)

    # Random classifier baseline (horizontal line at class prevalence = 1/num_classes)
    baseline = 1 / len(class_names)
    ax2.axhline(y=baseline, color='k', linestyle='--')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_optimization_results(
    study,
    figsize: tuple[float, float] = (8, 3),
    ylim: tuple[float, float] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot Optuna optimization history and hyperparameter importance.
    
    Args:
        study: Optuna study object with completed trials.
        figsize: Figure size (width, height).
        ylim: Optional y-axis limits for optimization history plot (min, max).
              Use this to zoom in on trends when there are outlier trials.
        
    Returns:
        Tuple of (figure, axes array).
    '''
    import optuna
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Optimization history
    axes[0].set_title('Optimization History')
    
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]

    # Plot individual trial results (markers only)
    axes[0].plot(trial_numbers, trial_values, 'ko', markersize=4, label='Trial results')
    
    # Calculate and plot running best (cumulative maximum)
    running_best = []
    best_so_far = float('-inf')
    for value in trial_values:
        best_so_far = max(best_so_far, value)
        running_best.append(best_so_far)
    
    axes[0].plot(trial_numbers, running_best, 'r-', linewidth=2, label=f'Best so far')
    
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].legend()
    
    # Set y-axis limits if provided
    if ylim is not None:
        axes[0].set_ylim(ylim)

    # Hyperparameter importance (if enough trials completed)
    axes[1].set_title('Hyperparameter Importance')
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) >= 5:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())
        
        axes[1].set_xlabel('Importance')
        axes[1].barh(params, values, color='black')

    else:
        axes[1].text(
            0.5, 0.5,
            'Not enough completed trials\nfor importance analysis', 
            ha='center', va='center', transform=axes[1].transAxes
        )

    plt.tight_layout()
    
    return fig, axes