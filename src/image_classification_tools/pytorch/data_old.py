'''Data loading and preprocessing functions for image classification datasets.

This module provides utilities for loading datasets (including CIFAR-10) and creating
PyTorch DataLoaders with support for custom transforms and device preloading.
'''

import json
import shutil
from pathlib import Path
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from tqdm import tqdm


def load_dataset(
    data_source: str | Path | type,
    transform: transforms.Compose,
    train: bool = True,
    **dataset_kwargs
) -> Dataset:
    '''Load a single dataset from a directory or PyTorch dataset class.
    
    This function provides a flexible interface for loading image classification datasets.
    It supports both PyTorch built-in datasets (CIFAR-10, CIFAR-100, MNIST, etc.) and
    custom datasets stored in directories following the ImageFolder structure.
    
    Args:
        data_source: Either a string or Path to a directory containing train/ or test/ subdirectory,
                    or a PyTorch dataset class (e.g., datasets.CIFAR10)
        transform: Transforms to apply to the data
        train: If True, load training data. If False, load test data (default: True)
        **dataset_kwargs: Additional keyword arguments passed to the dataset class
                         (e.g., root='data/pytorch/cifar10')
    
    Returns:
        Dataset object
    
    Raises:
        TypeError: If data_source is not a Path or a PyTorch dataset class
        ValueError: If directory-based dataset path does not exist
    
    Examples:
        # Load CIFAR-10 training data
        train_dataset = load_dataset(
            data_source=datasets.CIFAR10,
            transform=transform,
            train=True,
            root='data/cifar10'
        )
        
        # Load from ImageFolder
        train_dataset = load_dataset(
            data_source='data/my_dataset',
            transform=transform,
            train=True
        )
    '''
    
    if isinstance(data_source, (str, Path)):
        data_source = Path(data_source)

        # Directory-based dataset using ImageFolder
        subdir = 'train' if train else 'test'
        data_dir = data_source / subdir
        
        if not data_dir.exists():
            raise ValueError(f'{"Training" if train else "Test"} directory not found: {data_dir}')
        
        return datasets.ImageFolder(
            root=data_dir,
            transform=transform
        )
    
    elif isinstance(data_source, type) and issubclass(data_source, Dataset):

        # PyTorch dataset class (CIFAR-10, MNIST, etc.)
        # Try loading without download first, then download if not found
        try:
            return data_source(
                train=train,
                download=False,
                transform=transform,
                **dataset_kwargs
            )

        except (RuntimeError, FileNotFoundError):

            # Dataset not found on disk, download it
            return data_source(
                train=train,
                download=True,
                transform=transform,
                **dataset_kwargs
            )
    
    else:
        raise TypeError(
            f'data_source must be a Path or a PyTorch Dataset class, '
            f'got {type(data_source).__name__}'
        )


def prepare_splits(
    train_dataset: Dataset,
    test_dataset: Dataset | None = None,
    val_size: int = 10000,
    test_size: int | None = None
) -> Tuple[Dataset, Dataset, Dataset]:
    '''Split training dataset into train/val(/test) splits.
    
    The splitting behavior depends on whether a separate test dataset is provided:
    - If test_dataset is provided: Split train_dataset into train/val only (2-way split)
    - If test_dataset is None: Split train_dataset into train/val/test (3-way split)
    
    Args:
        train_dataset: Training dataset to split
        test_dataset: Test dataset. If None, test set will be split from train_dataset.
        val_size: Number of images to use for validation
        test_size: Number of images to reserve for testing when test_dataset is None.
                   Only used when test_dataset is None. If None when test_dataset is None,
                   raises ValueError.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    
    Examples:
        # 2-way split: Pass separate test set
        train_ds, val_ds, test_ds = prepare_splits(
            train_dataset=my_train_data,
            test_dataset=my_test_data,  # Use this for testing
            val_size=10000  # 10,000 images for validation
        )
        
        # 3-way split: No separate test set
        train_ds, val_ds, test_ds = prepare_splits(
            train_dataset=my_full_data,
            test_dataset=None,  # Will split test from train_dataset
            val_size=10000,  # 10,000 for validation
            test_size=5000  # 5,000 for testing
        )
    '''
    
    if test_dataset is not None:

        # 2-way split: train/val only, use provided test set
        total_size = len(train_dataset)
        
        if val_size >= total_size:
            raise ValueError(f'val_size ({val_size}) must be less than train_dataset size ({total_size})')
        
        indices = torch.randperm(total_size).tolist()
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_dataset_final = Subset(train_dataset, train_indices)
        val_dataset_final = Subset(train_dataset, val_indices)
        test_dataset_final = test_dataset
        
    else:

        # 3-way split: train/val/test all from train_dataset
        if test_size is None:
            raise ValueError('test_size must be provided when test_dataset is None')
        
        total_size = len(train_dataset)
        
        if val_size + test_size >= total_size:
            raise ValueError(
                f'val_size ({val_size}) + test_size ({test_size}) must be less than '
                f'train_dataset size ({total_size})'
            )
        
        indices = torch.randperm(total_size).tolist()
        
        val_indices = indices[:val_size]
        test_indices = indices[val_size:val_size + test_size]
        train_indices = indices[val_size + test_size:]
        
        train_dataset_final = Subset(train_dataset, train_indices)
        val_dataset_final = Subset(train_dataset, val_indices)
        test_dataset_final = Subset(train_dataset, test_indices)
    
    return train_dataset_final, val_dataset_final, test_dataset_final


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    preload_to_memory: bool = True,
    device: torch.device | None = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Create DataLoaders from prepared datasets with optional memory preloading.
    
    This function provides three memory management strategies:
    1. Lazy loading (preload_to_memory=False): Data stays on disk, loaded per batch
    2. CPU preloading (preload_to_memory=True, device=cpu): Entire dataset in RAM
    3. GPU preloading (preload_to_memory=True, device=cuda): Entire dataset in VRAM
    
    Args:
        train_dataset: Prepared training dataset
        val_dataset: Prepared validation dataset
        test_dataset: Prepared test dataset
        batch_size: Batch size for all DataLoaders
        shuffle_train: Whether to shuffle training data (default: True)
        num_workers: Number of subprocesses for data loading (default: 0 for single process).
                    Note: num_workers is ignored when preload_to_memory=True.
        preload_to_memory: If True, convert datasets to tensors and load into memory.
                          If False, keep as lazy-loading Dataset objects (default: True).
        device: Device to preload tensors onto. Only used if preload_to_memory=True.
               If None with preload_to_memory=True, defaults to CPU.
               Common values: torch.device('cpu'), torch.device('cuda')
        **kwargs: Additional keyword arguments passed to DataLoader
                 (e.g., pin_memory=True, persistent_workers=True)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Examples:
        # Strategy 1: Lazy loading (large datasets)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            num_workers=4,
            pin_memory=True
        )
        
        # Strategy 2: CPU preloading (medium datasets)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            preload_to_memory=True,
            device=torch.device('cpu')
        )
        
        # Strategy 3: GPU preloading (small datasets, fastest training)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            preload_to_memory=True,
            device=torch.device('cuda')
        )
    '''
    
    if preload_to_memory:
    
        # Preload datasets to memory
        if device is None:
            device = torch.device('cpu')
        
        # Load train data
        X_train = torch.stack([img for img, _ in train_dataset]).to(device)
        y_train = torch.tensor([label for _, label in train_dataset]).to(device)
        train_dataset_final = TensorDataset(X_train, y_train)
        
        # Load val data
        X_val = torch.stack([img for img, _ in val_dataset]).to(device)
        y_val = torch.tensor([label for _, label in val_dataset]).to(device)
        val_dataset_final = TensorDataset(X_val, y_val)
        
        # Load test data
        X_test = torch.stack([img for img, _ in test_dataset]).to(device)
        y_test = torch.tensor([label for _, label in test_dataset]).to(device)
        test_dataset_final = TensorDataset(X_test, y_test)
        
        # When preloading, num_workers should be 0
        num_workers = 0

    else:

        # Use datasets as-is for lazy loading
        train_dataset_final = train_dataset
        val_dataset_final = val_dataset
        test_dataset_final = test_dataset
    
    train_loader = DataLoader(
        train_dataset_final,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    return train_loader, val_loader, test_loader


def _augment_from_paths(path_batch: list, n_augmentations: int) -> int:
    """Worker: load images from paths, augment, and save.
    
    Args:
        path_batch: List of (img_path, output_dir) tuples
        n_augmentations: Number of augmented copies per image
        
    Returns:
        Number of images saved
    """
    from PIL import Image
    
    # Recreate transforms in worker (avoids pickling transform objects)
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
    
    count = 0
    for img_path, class_dir in path_batch:
        img_path = Path(img_path)
        class_dir = Path(class_dir)
        img = Image.open(img_path)
        
        # Extract index from filename (img_000001_orig.png -> 000001)
        idx = img_path.stem.split('_')[1]
        
        for aug_idx in range(n_augmentations):
            aug_img = augment(img)
            out_path = class_dir / f'img_{idx}_aug{aug_idx:02d}.png'
            aug_img.save(out_path)
            count += 1
    
    return count


def generate_preaugmented_dataset(
    dataset: Dataset,
    output_dir: str | Path,
    n_augmentations: int = 5,
    class_names: list[str] | None = None,
    num_workers: int = 8,
) -> Path:
    '''Generate pre-augmented dataset saved as ImageFolder format.
    
    Saves augmented images as PNG files in ImageFolder structure:
        output_dir/train/class_name/img_XXXXXX_augYY.png
    
    The output can be loaded directly with load_dataset():
        aug_dataset = load_dataset(output_dir, transform=transform, train=True)
    
    Args:
        dataset: Dataset with PIL images (no transform applied). Use training subset only!
        output_dir: Where to save the ImageFolder structure
        n_augmentations: Number of augmented copies per image (default: 5)
        class_names: Optional list of class names for subdirectories. 
                     If None, uses 'class_0', 'class_1', etc.
        num_workers: Number of parallel workers for augmentation (default: 8)
    
    Returns:
        Path to output directory
        
    Example:
        >>> # Generate augmented training data
        >>> generate_preaugmented_dataset(
        ...     train_subset, 
        ...     'data/augmented_cifar10',
        ...     n_augmentations=5,
        ...     class_names=['airplane', 'automobile', ...]
        ... )
        >>> # Load with existing pipeline
        >>> aug_train = load_dataset('data/augmented_cifar10', transform, train=True)
    '''
    from concurrent.futures import ProcessPoolExecutor, as_completed

    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    
    # Always regenerate to prevent data leakage
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    
    # Get unique classes and create directories
    print('Scanning dataset for classes...')
    all_labels = set()
    for _, label in dataset:
        all_labels.add(label if isinstance(label, int) else label.item())
    unique_classes = sorted(all_labels)
    
    # Create class directories
    class_dirs = {}
    for class_idx in unique_classes:
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f'class_{class_idx}'
        class_dir = train_dir / class_name
        class_dir.mkdir(exist_ok=True)
        class_dirs[class_idx] = class_dir
    
    print(f'Found {len(unique_classes)} classes')
    
    # Phase 1: Save original images to disk (collect paths for workers)
    print(f'Saving {len(dataset)} original images...')
    path_list = []  # (img_path, class_dir) tuples
    
    for idx in tqdm(range(len(dataset)), desc='Saving originals'):
        img, label = dataset[idx]
        label_val = label if isinstance(label, int) else label.item()
        class_dir = class_dirs[label_val]
        
        img_path = class_dir / f'img_{idx:06d}_orig.png'
        img.save(img_path)
        path_list.append((str(img_path), str(class_dir)))
    
    # Phase 2: Parallel augmentation from saved paths
    print(f'Generating {len(dataset) * n_augmentations} augmented images with {num_workers} workers...')
    
    # Split paths into batches for workers
    batch_size = max(1, len(path_list) // num_workers)
    batches = [path_list[i:i + batch_size] for i in range(0, len(path_list), batch_size)]
    
    total_augmented = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_augment_from_paths, batch, n_augmentations)
            for batch in batches
        ]
        for future in as_completed(futures):
            total_augmented += future.result()
    
    total_saved = len(dataset) + total_augmented
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'n_augmentations': n_augmentations,
            'original_size': len(dataset),
            'augmented_size': total_saved,
            'class_names': class_names or [f'class_{i}' for i in unique_classes],
        }, f, indent=2)
    
    print(f'Done: {total_saved} images saved to {output_dir}')
    return output_dir
