'''Project level global variables and configuration settings'''

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import optuna
import torch
from torchvision import transforms

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Suppress Optuna info messages (show only warnings and errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
MODELS_DIR = Path('../models/pytorch')
RESULTS_DIR = Path('../data/pytorch/performance_results')
DATA_DIR = Path('../data/pytorch/cifar10')
AUGMENTED_DATA_DIR = Path('../data/pytorch/augmented_cifar10')
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

# ============================================================================
# Data Pipeline Configuration
# ============================================================================

# Default hyperparameters for data loading
VAL_SIZE = 10000
BATCH_SIZE = 128

# Transform presets for different use cases
GRAYSCALE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

RGB_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Augmentation transform presets
PIL_AUGMENTATIONS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

TENSOR_AUGMENTATIONS = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

# RGB transforms for training (eval transform is same as RGB_TRANSFORM)
RGB_TRAIN_TRANSFORM = RGB_TRANSFORM  # Augmentation handled separately via DataPipeline
RGB_EVAL_TRANSFORM = RGB_TRANSFORM

# Cache key for pregenerated augmentation
AUGMENTATION_CACHE_KEY = 'cifar10_standard_aug_v1'
