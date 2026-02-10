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

# Suppress Optuna info messages (show only warnings and errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
MODELS_DIR = Path('../models/pytorch')
RESULTS_DIR = Path('../data/pytorch/performance_results')
DATA_DIR = Path('../data/pytorch/cifar10')

# Make sure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 class names in class order
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]