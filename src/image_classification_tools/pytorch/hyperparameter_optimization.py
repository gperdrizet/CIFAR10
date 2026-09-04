'''Hyperparameter optimization utilities for CNN models using Optuna.

This module provides functions for running hyperparameter optimization with Optuna.
Model architecture definition should be done in the notebook/experiment code.

Optuna is an optional dependency. The functions in this module raise a clear
ImportError if called without optuna installed, but importing this module
(and the rest of the package) works fine without it.
'''

from __future__ import annotations

from typing import Callable

try:
    import optuna
except ImportError:
    optuna = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from image_classification_tools.pytorch.data import DataPipeline


def _require_optuna() -> None:
    '''Raise a clear error if optuna is not installed.'''
    if optuna is None:
        raise ImportError(
            "optuna is required for hyperparameter optimization. "
            "Install it with: pip install optuna"
        )


class TrialFailedError(Exception):
    '''Exception raised when an Optuna trial fails (e.g., OOM, dimension collapse).
    
    Use with study.optimize(catch=(TrialFailedError,)) to mark trials as FAIL
    instead of crashing the optimization.
    '''
    pass


class MockTrial:
    '''Mock Optuna trial that returns fixed hyperparameters.
    
    Use this to recreate models with specific hyperparameters after optimization.
    Instead of sampling new values, it returns pre-determined values from a dictionary.
    
    Example:
        >>> best_params = study.best_trial.params
        >>> mock_trial = MockTrial(best_params)
        >>> model = create_model(mock_trial, num_classes=10, in_channels=3)
    '''
    
    def __init__(self, params: dict):
        '''Initialize with fixed parameter values.
        
        Args:
            params: Dictionary of hyperparameter names and values
        '''
        self.params = params
    
    def suggest_int(self, name: str, *args, **kwargs) -> int:
        '''Return the fixed integer parameter value.'''
        return self.params[name]
    
    def suggest_categorical(self, name: str, *args, **kwargs):
        '''Return the fixed categorical parameter value.'''
        return self.params[name]
    
    def suggest_float(self, name: str, *args, **kwargs) -> float:
        '''Return the fixed float parameter value.'''
        return self.params[name]


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
    _require_optuna()

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
    model_factory: Callable[[optuna.Trial, int, int], nn.Module],
    data_source,
    data_dir,
    train_transform,
    eval_transform,
    n_epochs: int,
    num_classes: int,
    in_channels: int = 3,
    val_size: int = 10000,
    search_space: dict = None,
    early_stopping_patience: int = None,
    min_delta: float = 0.0
) -> Callable[[optuna.Trial], float]:
    '''Create an Optuna objective function for CNN hyperparameter optimization.
    
    This factory function creates a closure that captures the data loading parameters
    and training configuration, returning an objective function suitable for Optuna.
    The objective function automatically selects GPUs in round-robin fashion when
    multiple GPUs are available and n_jobs > 1.
    
    Args:
        model_factory: Function that creates a model given (trial, num_classes, in_channels)
        data_source: Dataset class (e.g., datasets.CIFAR10, datasets.MNIST)
        data_dir: Directory containing training data
        train_transform: Transform to apply to training data
        eval_transform: Transform to apply to validation data
        n_epochs: Number of epochs per trial
        num_classes: Number of output classes (required, e.g., 10 for CIFAR-10)
        in_channels: Number of input channels (default: 3 for RGB images, 1 for grayscale)
        val_size: Number of validation samples (default: 10000)
        search_space: Dictionary defining hyperparameter search space (default: None)
        early_stopping_patience: Number of epochs to wait before stopping if validation loss doesn't improve (None to disable, default: None)
        min_delta: Minimum change in validation loss to qualify as an improvement (default: 0.0)
    
    Returns:
        Objective function for optuna.Study.optimize()
    
    Example:
        >>> def my_model_factory(trial, num_classes, in_channels):
        ...     n_blocks = trial.suggest_int('n_blocks', 3, 10)
        ...     return create_cnn(n_blocks, num_classes, in_channels)
        >>> 
        >>> objective = create_objective(
        ...     model_factory=my_model_factory,
        ...     data_source=datasets.CIFAR10,
        ...     data_dir='data/', 
        ...     train_transform=transform,
        ...     eval_transform=transform,
        ...     n_epochs=50, 
        ...     num_classes=10,
        ...     early_stopping_patience=5
        ... )
        >>> study = optuna.create_study(direction='maximize')
        >>> study.optimize(objective, n_trials=100, n_jobs=4)
    '''
    _require_optuna()

    if search_space == None:
        return None
    
    def objective(trial: optuna.Trial) -> float:
        '''Optuna objective function for CNN hyperparameter optimization.'''
        
        # Auto-select GPU device for this trial (round-robin across available GPUs)
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_id = trial.number % n_gpus
            device = torch.device(f'cuda:{gpu_id}')
            # Preload to the specific GPU assigned to this trial
            preload_device = f'cuda:{gpu_id}'
        else:
            device = torch.device('cpu')
            preload_device = 'cpu'
        
        # Suggest hyperparameters from search space
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
        
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
        
        # Create data pipeline with suggested batch size
        loaders = DataPipeline(
            data_source=data_source,
            data_dir=data_dir,
            split='train/val',
            val_size=val_size,
            batch_size=batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            preload=preload_device
        ).get_loaders()
        
        train_loader = loaders.train
        val_loader = loaders.val
        
        # Wrap model creation and training in try/except to catch OOM errors
        try:
            # Create model using the provided model factory
            model = model_factory(trial, num_classes, in_channels).to(device)
            
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
