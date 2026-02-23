'''Project level global variables and configuration settings'''

# Standard library imports
from pathlib import Path
# import os

# Third-party imports
import numpy as np
import optuna
import torch
from torchvision import transforms
from dotenv import load_dotenv

# Load environment variables from .env if it exists
env_path = Path(__file__).parent.parent / '.env'

if env_path.exists():
    load_dotenv(env_path)

# Set random seeds for reproducibility
SEED = 315
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
HISTORY_DIR = Path('../data/pytorch/performance_results')  # Directory for training history JSON files
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

# CIFAR-10 has a standard train/test split of 50k/10k images.
# We will further split the training set into train/val.
SPLIT = 'train/val/test'

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

# Augmented dataset name for caching
AUGMENTED_DATASET_NAME = 'cifar10_standard_aug_v1'

# ============================================================================
# Model Filenames
# ============================================================================


# Model filenames for each notebook
MODEL_DNN = '01-dnn.pth'
MODEL_CNN = '02-cnn.pth'
MODEL_RGB_CNN = '03-rgb_cnn.pth'
MODEL_OPTIMIZED_CNN = '04-optimized_cnn.pth'
MODEL_TRAINING_OPTIMIZED_CNN = '05-training_optimized_cnn.pth'
MODEL_AUGMENTED_CNN = '06-augmented_cnn.pth'
MODEL_RESNET50 = '07-resnet50.pth'

# Training history filenames for each notebook (JSON format)
HISTORY_DNN = '01-dnn_history.json'
HISTORY_CNN = '02-cnn_history.json'
HISTORY_RGB_CNN = '03-rgb_cnn_history.json'
HISTORY_OPTIMIZED_CNN = '04-optimized_cnn_history.json'
HISTORY_TRAINING_OPTIMIZED_CNN = '05-training_optimized_cnn_history.json'
HISTORY_AUGMENTED_CNN = '06-augmented_cnn_history.json'
HISTORY_RESNET50 = '07-resnet50_history.json'

# Results filenames for each notebook
RESULTS_DNN = '01-dnn_results.pkl'
RESULTS_CNN = '02-cnn_results.pkl'
RESULTS_RGB_CNN = '03-rgb_cnn_results.pkl'
RESULTS_OPTIMIZED_CNN = '04-architecture_optimized_cnn_results.pkl'
RESULTS_TRAINING_OPTIMIZED_CNN = '05-training_optimized_cnn_results.pkl'
RESULTS_AUGMENTED_CNN = '06-augmented_cnn_results.pkl'
RESULTS_RESNET50 = '07-resnet50_results.pkl'
