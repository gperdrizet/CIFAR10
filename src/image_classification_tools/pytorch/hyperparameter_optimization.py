'''Hyperparameter optimization utilities for CNN models using Optuna.

This module provides functions for building configurable CNN architectures
and running hyperparameter optimization with Optuna.
'''

from typing import Callable

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from image_classification_tools.pytorch.data import (
    load_dataset, prepare_splits, create_dataloaders
)


def create_cnn(
    n_conv_blocks: int,
    initial_filters: int,
    n_fc_layers: int,
    conv_dropout_rate: float,
    fc_dropout_rate: float,
    num_classes: int,
    in_channels: int = 3
) -> nn.Sequential:
    '''Create a CNN with configurable architecture.
    
    This function builds a flexible CNN architecture with conv blocks that
    progressively double filters (initial_filters -> 2x -> 4x -> 8x, etc.).
    Each conv block contains: 2 Conv layers + BatchNorm + ReLU + MaxPool + Dropout.
    Uses adaptive pooling before classifier to handle variable spatial dimensions.
    
    Args:
        n_conv_blocks: Number of convolutional blocks (1-5)
        initial_filters: Number of filters in first conv block (doubles each block)
        n_fc_layers: Number of fully connected layers before output (1-4)
        conv_dropout_rate: Dropout probability after convolutional blocks
        fc_dropout_rate: Dropout probability in fully connected layers
        num_classes: Number of output classes (required)
        in_channels: Number of input channels (default: 3 for RGB images)
    
    Returns:
        nn.Sequential model
    '''

    layers = []
    current_channels = in_channels
    
    # Convolutional blocks
    for block_idx in range(n_conv_blocks):
        out_channels = initial_filters * (2 ** block_idx)
        
        # First conv in block
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Second conv in block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Pooling and dropout
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout(conv_dropout_rate))
        
        current_channels = out_channels
    
    # Classifier with adaptive pooling
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    
    # Generate FC layer sizes by halving from current_channels
    fc_sizes = []
    current_fc_size = current_channels // 2

    for _ in range(n_fc_layers):

        fc_sizes.append(max(32, current_fc_size))  # Minimum 32 units
        current_fc_size //= 2
    
    # Add FC layers
    in_features = current_channels

    for fc_size in fc_sizes:

        layers.append(nn.Linear(in_features, fc_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(fc_dropout_rate))
        in_features = fc_size
    
    # Output layer
    layers.append(nn.Linear(in_features, num_classes))
    
    return nn.Sequential(*layers)


def train_trial(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    trial: optuna.Trial
) -> float:
    '''Train a model for a single Optuna trial with pruning support.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer for training
        criterion: Loss function
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of epochs to train
        trial: Optuna trial object for reporting and pruning
    
    Returns:
        Best validation accuracy achieved during training
    '''
    best_val_accuracy = 0.0
    
    for epoch in range(n_epochs):

        # Training phase
        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        
        # Report intermediate value for pruning
        trial.report(val_accuracy, epoch)
        
        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_accuracy


def create_objective(
    data_dir,
    transform,
    n_epochs: int,
    device: torch.device,
    num_classes: int,
    in_channels: int = 3,
    search_space: dict = None
) -> Callable[[optuna.Trial], float]:
    '''Create an Optuna objective function for CNN hyperparameter optimization.
    
    This factory function creates a closure that captures the data loading parameters
    and training configuration, returning an objective function suitable for Optuna.
    
    Args:
        data_dir: Directory containing training data
        transform: Transform to apply to both training and validation data
        n_epochs: Number of epochs per trial
        device: Device to train on (cuda or cpu)
        num_classes: Number of output classes (required, e.g., 10 for CIFAR-10)
        in_channels: Number of input channels (default: 3 for RGB images, 1 for grayscale)
        search_space: Dictionary defining hyperparameter search space (default: None)
    
    Returns:
        Objective function for optuna.Study.optimize()
    
    Example:
        >>> objective = create_objective(
        ...     data_dir='data/', 
        ...     transform=transform,
        ...     n_epochs=50, 
        ...     device=device,
        ...     num_classes=10
        ... )
        >>> study = optuna.create_study(direction='maximize')
        >>> study.optimize(objective, n_trials=100)
    '''

    if search_space == None:
        return None
    
    def objective(trial: optuna.Trial) -> float:
        '''Optuna objective function for CNN hyperparameter optimization.'''
        
        # Suggest hyperparameters from search space
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
        n_conv_blocks = trial.suggest_int('n_conv_blocks', *search_space['n_conv_blocks'])
        initial_filters = trial.suggest_categorical('initial_filters', search_space['initial_filters'])
        n_fc_layers = trial.suggest_int('n_fc_layers', *search_space['n_fc_layers'])
        conv_dropout_rate = trial.suggest_float('conv_dropout_rate', *search_space['conv_dropout_rate'])
        fc_dropout_rate = trial.suggest_float('fc_dropout_rate', *search_space['fc_dropout_rate'])
        
        # Handle learning rate with optional log scale
        lr_params = search_space['learning_rate']
        learning_rate = trial.suggest_float(
            'learning_rate', lr_params[0], lr_params[1], log=(lr_params[2] == 'log' if len(lr_params) > 2 else False)
        )
        
        # Optimizer (optional, defaults to Adam)
        if 'optimizer' in search_space:
            optimizer_name = trial.suggest_categorical('optimizer', search_space['optimizer'])
        else:
            optimizer_name = 'Adam'
        
        # Weight decay (optional, defaults to 0)
        if 'weight_decay' in search_space:
            wd_params = search_space['weight_decay']
            weight_decay = trial.suggest_float(
                'weight_decay', wd_params[0], wd_params[1], log=(wd_params[2] == 'log' if len(wd_params) > 2 else False)
            )
        else:
            weight_decay = 0.0
        
        # Create data loaders with suggested batch size
        # Load datasets
        train_dataset = load_dataset(
            data_source=datasets.CIFAR10,
            transform=transform,
            train=True,
            root=data_dir
        )
        
        test_dataset = load_dataset(
            data_source=datasets.CIFAR10,
            transform=transform,
            train=False,
            root=data_dir
        )
        
        # Prepare splits
        train_dataset, val_dataset, _ = prepare_splits(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_size=10000
        )
        
        # Create dataloaders with memory preloading
        train_loader, val_loader, _ = create_dataloaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=batch_size,
            preload_to_memory=(device is not None),
            device=device
        )
        
        # Wrap model creation and training in try/except to catch OOM errors
        try:
            # Create model with suggested architecture
            model = create_cnn(
                n_conv_blocks=n_conv_blocks,
                initial_filters=initial_filters,
                n_fc_layers=n_fc_layers,
                conv_dropout_rate=conv_dropout_rate,
                fc_dropout_rate=fc_dropout_rate,
                num_classes=num_classes,
                in_channels=in_channels
            ).to(device)
            
            # Define optimizer
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            elif optimizer_name == 'SGD':
                if 'sgd_momentum' in search_space:
                    momentum = trial.suggest_float('sgd_momentum', *search_space['sgd_momentum'])
                else:
                    momentum = 0.9
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                                    momentum=momentum, weight_decay=weight_decay)
            
            else:  # RMSprop
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            criterion = nn.CrossEntropyLoss()
            
            # Train model and return best validation accuracy
            return train_trial(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                trial=trial
            )

        except RuntimeError as e:
            # Catch architecture errors (e.g., dimension mismatches)
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f'RuntimeError with params: {trial.params} - {str(e)}')
        
        except torch.cuda.OutOfMemoryError:
            # Clear CUDA cache and skip this trial
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f'CUDA OOM with params: {trial.params}')
    
    return objective
