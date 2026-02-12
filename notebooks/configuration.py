'''Project level global variables and configuration setting'''

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import optuna
import torch

# Set random seeds for reproducibility
torch.manual_seed(315)
np.random.seed(315)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Suppress Optuna info messages (show only warnings and errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
MODELS_DIR = Path('../models/pytorch')
RESULTS_DIR = Path('../data/pytorch/performance_results')
DATA_DIR = Path('../data/pytorch/cifar10')
OPTUNA_DB_PATH = Path('../data/pytorch/cnn_optimization.db')
DOCS_DIR = Path('../docs')

# Make sure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DOCS_DIR / 'assets').mkdir(parents=True, exist_ok=True)

# Optuna storage URL (use absolute path for SQLite)
OPTUNA_STORAGE_URL = f'sqlite:///{OPTUNA_DB_PATH.resolve()}'

# CIFAR-10 class names in class order
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]