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


class TrialFailedError(Exception):
    '''Exception raised when an Optuna trial fails (e.g., OOM, dimension collapse).
    
    Use with study.optimize(catch=(TrialFailedError,)) to mark trials as FAIL
    instead of crashing the optimization.
    '''
    pass


def create_cnn(
    n_conv_blocks: int,
    initial_filters: int,
    n_fc_layers: int,
    conv_dropout_rate: float,
    fc_dropout_rate: float,
    num_classes: int,
    in_channels: int = 3,
    pool_frequency: int = 2
) -> nn.Sequential:
    '''Create a CNN with configurable architecture.
    
    This function builds a flexible CNN architecture with conv blocks that
    progressively double filters every 2 blocks for deeper networks.
    Each conv block contains: 2 Conv layers + BatchNorm + ReLU + Conditional MaxPool + Dropout.
    MaxPooling frequency is configurable to support deeper architectures without
    spatial dimension collapse. For 32x32 inputs:
      - pool_frequency=2: up to 10 blocks (5 pools)
      - pool_frequency=3: up to 15 blocks (5 pools)
      - pool_frequency=4: up to 20 blocks (5 pools)
    Uses adaptive pooling before classifier to handle variable spatial dimensions.
    
    Args:
        n_conv_blocks: Number of convolutional blocks
        initial_filters: Number of filters in first conv block (doubles every 2 blocks)
        n_fc_layers: Number of fully connected layers before output (1-5)
        conv_dropout_rate: Dropout probability after convolutional blocks
        fc_dropout_rate: Dropout probability in fully connected layers
        num_classes: Number of output classes (required)
        in_channels: Number of input channels (default: 3 for RGB images)
        pool_frequency: Apply MaxPool every N blocks (default: 2)
    
    Returns:
        nn.Sequential model
    '''

    layers = []
    current_channels = in_channels
    
    # Convolutional blocks
    for block_idx in range(n_conv_blocks):
        # Double filters every 2 blocks (slower growth for deeper networks)
        out_channels = initial_filters * (2 ** (block_idx // 2))
        
        # First conv in block
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Second conv in block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        # Pool every N blocks, or after the last block (enables deeper architectures)
        if (block_idx + 1) % pool_frequency == 0 or (block_idx + 1) == n_conv_blocks:
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
    trial: optuna.Trial,
    early_stopping_patience: int = None,
    min_delta: float = 0.0,
    scheduler = None
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
        early_stopping_patience: Number of epochs to wait before stopping if validation loss doesn't improve (None to disable)
        min_delta: Minimum change in validation loss to qualify as an improvement (default: 0.0)
        scheduler: Optional learning rate scheduler (e.g., CosineAnnealingLR, StepLR)
    
    Returns:
        Best validation accuracy achieved during training
    '''
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):

        # Training phase
        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Step scheduler at end of epoch (if provided)
        if scheduler is not None:
            scheduler.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        
        # Early stopping based on validation loss
        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                # Mark trial as stopped early for dashboard visibility
                trial.set_user_attr('stopped_early', True)
                trial.set_user_attr('stopped_at_epoch', epoch + 1)
                trial.set_user_attr('total_epochs_planned', n_epochs)
                break
        
        # Report intermediate value for pruning (use loss for pruner, lower is better)
        trial.report(avg_val_loss, epoch)
        
        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Mark if trial completed all epochs without early stopping
    if early_stopping_patience is not None and patience_counter < early_stopping_patience:
        trial.set_user_attr('stopped_early', False)
        trial.set_user_attr('completed_epochs', n_epochs)
    
    return best_val_accuracy


def create_objective(
    data_dir,
    transform,
    n_epochs: int,
    device: torch.device,
    num_classes: int,
    in_channels: int = 3,
    search_space: dict = None,
    early_stopping_patience: int = None,
    min_delta: float = 0.0
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
        early_stopping_patience: Number of epochs to wait before stopping if validation loss doesn't improve (None to disable, default: None)
        min_delta: Minimum change in validation loss to qualify as an improvement (default: 0.0)
    
    Returns:
        Objective function for optuna.Study.optimize()
    
    Example:
        >>> objective = create_objective(
        ...     data_dir='data/', 
        ...     transform=transform,
        ...     n_epochs=50, 
        ...     device=device,
        ...     num_classes=10,
        ...     early_stopping_patience=5
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
        
        # Pool frequency (optional, defaults to 2)
        if 'pool_frequency' in search_space:
            pool_frequency = trial.suggest_categorical('pool_frequency', search_space['pool_frequency'])
        else:
            pool_frequency = 2
        
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
                in_channels=in_channels,
                pool_frequency=pool_frequency
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
                trial=trial,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta
            )

        except torch.cuda.OutOfMemoryError as e:
            # Catch OOM first (it's a subclass of RuntimeError)
            torch.cuda.empty_cache()
            trial.set_user_attr('failure_reason', 'cuda_oom')
            trial.set_user_attr('error_message', str(e))
            # Raise to mark trial as FAIL (requires catch= in study.optimize)
            raise TrialFailedError(f'CUDA OOM: {e}')
        
        except RuntimeError as e:
            # Catch architecture errors (e.g., dimension collapse, layer mismatches)
            error_msg = str(e)
            torch.cuda.empty_cache()
            
            # Check if this is a dimension/OOM error
            if 'Output size is too small' in error_msg or 'Calculated output size' in error_msg:
                # Report to trial as user attribute for debugging
                trial.set_user_attr('failure_reason', 'dimension_collapse')
                trial.set_user_attr('error_message', error_msg)
                raise TrialFailedError(f'Dimension collapse: {error_msg}')
            elif 'out of memory' in error_msg.lower():
                # Sometimes OOM is wrapped in RuntimeError
                trial.set_user_attr('failure_reason', 'cuda_oom')
                trial.set_user_attr('error_message', error_msg)
                raise TrialFailedError(f'CUDA OOM: {error_msg}')
            else:
                # Other RuntimeErrors should still crash (unexpected issues)
                raise RuntimeError(f'RuntimeError with params: {trial.params} - {error_msg}')
    
    return objective
