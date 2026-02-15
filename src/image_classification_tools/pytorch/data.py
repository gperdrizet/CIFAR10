'''Data loading and preprocessing for image classification datasets.

This module provides a simplified, unified API for loading datasets, splitting them,
and creating PyTorch DataLoaders with support for GPU/CPU preloading and data augmentation.
'''

import json
import shutil
import warnings
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class AugmentationStrategy(Enum):
    """Strategy for applying data augmentation."""
    NONE = 'none'
    ON_THE_FLY = 'on_the_fly'
    PREGENERATED = 'pregenerated'


@dataclass(frozen=True)
class DataLoaders:
    """Container for train/val/test DataLoaders with convenience methods.
    
    Attributes:
        train: Training DataLoader (or None if not in split)
        val: Validation DataLoader (or None if not in split)
        test: Test DataLoader (or None if not in split)
        batch_size: Batch size used for all loaders
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        device: Device where data is loaded ('cpu', 'cuda', or None for lazy)
    """
    train: Optional[DataLoader]
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    batch_size: int
    train_size: int
    val_size: int
    test_size: int
    device: Optional[str]
    
    def get_batch_sizes(self) -> Dict[str, int]:
        """Get batch sizes for each split.
        
        Returns:
            Dictionary with batch sizes for available splits
        """
        result = {}
        if self.train is not None:
            result['train'] = self.batch_size
        if self.val is not None:
            result['val'] = self.batch_size
        if self.test is not None:
            result['test'] = self.batch_size
        return result
    
    def total_samples(self) -> Dict[str, int]:
        """Get total sample counts for each split.
        
        Returns:
            Dictionary with sample counts for available splits
        """
        result = {}
        if self.train is not None:
            result['train'] = self.train_size
        if self.val is not None:
            result['val'] = self.val_size
        if self.test is not None:
            result['test'] = self.test_size
        return result
    
    def split_info(self) -> str:
        """Get formatted string with split information.
        
        Returns:
            Human-readable string like "Train: 40,000 | Val: 10,000 | Test: 10,000"
        """
        parts = []
        if self.train is not None:
            parts.append(f"Train: {self.train_size:,}")
        if self.val is not None:
            parts.append(f"Val: {self.val_size:,}")
        if self.test is not None:
            parts.append(f"Test: {self.test_size:,}")
        return " | ".join(parts)
    
    def memory_estimate(self) -> str:
        """Estimate memory usage for preloaded data.
        
        Returns:
            Human-readable string like "Estimated memory: 2.3 GB"
            Returns "N/A (lazy loading)" if data not preloaded
        """
        if self.device is None:
            return "N/A (lazy loading)"
        
        # Estimate: assume 32x32x3 RGB images with float32 (4 bytes per value)
        bytes_per_image = 32 * 32 * 3 * 4
        total_samples = self.train_size + self.val_size + self.test_size
        total_bytes = total_samples * bytes_per_image
        total_gb = total_bytes / (1024 ** 3)
        
        return f"Estimated memory: {total_gb:.2f} GB"


class DataPipeline:
    """Unified data loading pipeline with auto-detection and intelligent splitting.
    
    This class replaces the old 3-step workflow (load → split → create_dataloaders) with
    a single, outcome-based API. User specifies desired splits, pipeline auto-detects
    source capabilities and performs minimal operations to achieve the outcome.
    
    Examples:
        # Basic usage with GPU preloading
        >>> loaders = DataPipeline(
        ...     data_source=datasets.CIFAR10,
        ...     split='train/val/test',
        ...     train_transform=my_transform,
        ...     eval_transform=my_transform,
        ...     preload='gpu'
        ... ).get_loaders()
        >>> train_loader = loaders.train
        
        # With pregenerated augmentation
        >>> loaders = DataPipeline(
        ...     data_source=datasets.CIFAR10,
        ...     split='train/val/test',
        ...     train_transform=eval_transform,
        ...     eval_transform=eval_transform,
        ...     augmentation='pregenerated',
        ...     pil_augmentations=my_pil_augs,
        ...     cache_key='cifar10_strong_aug_v1',
        ...     preload='cpu'
        ... ).get_loaders()
    """
    
    def __init__(
        self,
        data_source: type | str | Path,
        split: str,
        root: Optional[str | Path] = None,
        train_transform: Optional[transforms.Compose] = None,
        eval_transform: Optional[transforms.Compose] = None,
        preload: Optional[str] = None,
        augmentation: str | AugmentationStrategy = 'none',
        pil_augmentations: Optional[transforms.Compose] = None,
        tensor_augmentations: Optional[transforms.Compose] = None,
        cache_key: Optional[str] = None,
        val_size: int = 10000,
        test_size: int = 10000,
        batch_size: int = 128,
        num_workers: int = 0,
        seed: int = 42,
        shuffle_train: bool = True,
        force_regenerate: bool = False,
        **loader_kwargs
    ):
        """Initialize DataPipeline.
        
        Args:
            data_source: PyTorch dataset class (e.g., datasets.CIFAR10) or path to data directory
            split: Desired split outcome. Must be one of: 'train', 'train/val', 'train/test', 'train/val/test'
            root: Root directory for data (required for PyTorch datasets)
            train_transform: Transform for training data (required, will fail if None)
            eval_transform: Transform for validation/test data (required, will fail if None)
            preload: Loading strategy: 'gpu', 'cpu', or None for lazy loading from disk
            augmentation: Augmentation strategy: 'none', 'on_the_fly', or 'pregenerated'
            pil_augmentations: Custom PIL augmentation transforms (flip, rotate, etc.)
            tensor_augmentations: Custom tensor augmentation transforms (blur, erasing, etc.)
            cache_key: Required string for pregenerated augmentation caching
            val_size: Number of validation samples (default: 10000)
            test_size: Number of test samples for 3-way splits (default: 10000)
            batch_size: Batch size for all loaders (default: 128)
            num_workers: Number of workers for DataLoader and augmentation (default: 0)
            seed: Random seed for reproducible splits (default: 42)
            shuffle_train: Whether to shuffle training data (default: True)
            force_regenerate: Force regeneration of cached augmented data (default: False)
            **loader_kwargs: Additional arguments passed to DataLoader
        
        Raises:
            ValueError: If transforms are None, split format invalid, or incompatible config
        """
        self.data_source = data_source
        self.split_str = split
        self.root = Path(root) if root else None
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.preload = preload
        self.augmentation = AugmentationStrategy(augmentation) if isinstance(augmentation, str) else augmentation
        self.pil_augmentations = pil_augmentations
        self.tensor_augmentations = tensor_augmentations
        self.cache_key = cache_key
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle_train = shuffle_train
        self.force_regenerate = force_regenerate
        self.loader_kwargs = loader_kwargs
        
        # Validate configuration
        self._validate_config()
        
        # Parse split string
        self.splits_needed = self._parse_split(split)
        
        # Detect source capabilities
        self.source_splits = self._detect_source_splits()
        
        # Plan operations
        self.split_plan = self._plan_split_operations()
    
    def _validate_config(self):
        """Validate configuration and auto-correct incompatible settings with warnings."""
        # Check transforms are provided
        if self.train_transform is None:
            raise ValueError(
                "train_transform is required. Please provide a transforms.Compose object."
            )
        if self.eval_transform is None:
            raise ValueError(
                "eval_transform is required. Please provide a transforms.Compose object."
            )
        
        # Validate preload
        if self.preload is not None and self.preload not in ['gpu', 'cpu']:
            raise ValueError(
                f"preload must be 'gpu', 'cpu', or None, got: {self.preload}"
            )
        
        # Check augmentation + preload compatibility
        if self.preload in ['gpu', 'cpu'] and self.augmentation == AugmentationStrategy.ON_THE_FLY:
            warnings.warn(
                "On-the-fly augmentation is not compatible with preloading. "
                "Auto-correcting to 'pregenerated' augmentation.",
                UserWarning
            )
            self.augmentation = AugmentationStrategy.PREGENERATED
        
        # Check cache_key for pregenerated augmentation
        if self.augmentation == AugmentationStrategy.PREGENERATED and self.cache_key is None:
            raise ValueError(
                "cache_key is required when augmentation='pregenerated'. "
                "Provide a unique string identifier for caching (e.g., 'cifar10_aug_v1')."
            )
        
        # Warn about inefficient pregenerated + lazy loading
        if self.preload is None and self.augmentation == AugmentationStrategy.PREGENERATED:
            warnings.warn(
                "Pregenerated augmentation with lazy loading is inefficient. "
                "Consider using preload='cpu' or 'gpu' for better performance.",
                UserWarning
            )
    
    def _parse_split(self, split_str: str) -> List[str]:
        """Parse and validate split string.
        
        Args:
            split_str: Split specification like 'train/val/test'
        
        Returns:
            List of split names like ['train', 'val', 'test']
        
        Raises:
            ValueError: If split format is invalid
        """
        import re
        
        # Validate format: must be train with optional /val and/or /test
        valid_patterns = [
            r'^train$',
            r'^train/val$',
            r'^train/test$',
            r'^train/val/test$'
        ]
        
        if not any(re.match(pattern, split_str) for pattern in valid_patterns):
            raise ValueError(
                f"Invalid split format: '{split_str}'. "
                "Must be one of: 'train', 'train/val', 'train/test', 'train/val/test'"
            )
        
        return split_str.split('/')
    
    def _detect_source_splits(self) -> str:
        """Detect if data source has pre-made splits.
        
        Returns:
            'train+test' if source is PyTorch dataset with separate train/test
            'single' if source is directory or single dataset
        """
        if isinstance(self.data_source, (str, Path)):
            # Directory-based source
            return 'single'
        
        # Check if it's a PyTorch dataset class with train parameter
        try:
            import inspect
            sig = inspect.signature(self.data_source.__init__)
            if 'train' in sig.parameters:
                return 'train+test'
        except (AttributeError, TypeError):
            pass
        
        return 'single'
    
    def _plan_split_operations(self) -> Dict[str, Any]:
        """Plan minimal split operations to achieve desired outcome.
        
        Returns:
            Dictionary describing operations needed
        """
        plan = {
            'load_train': False,
            'load_test': False,
            'split_train_into_train_val': False,
            'split_train_into_all': False,
            'val_size': self.val_size,
            'test_size': self.test_size
        }
        
        needs_val = 'val' in self.splits_needed
        needs_test = 'test' in self.splits_needed
        
        if self.source_splits == 'train+test':
            # Source has separate train and test
            plan['load_train'] = True
            
            if needs_test:
                plan['load_test'] = True
            
            if needs_val:
                # Need to split train into train/val
                plan['split_train_into_train_val'] = True
        
        else:
            # Single dataset source
            plan['load_train'] = True
            
            if needs_val and needs_test:
                # 3-way split
                plan['split_train_into_all'] = True
            elif needs_val:
                # 2-way split: train/val
                plan['split_train_into_train_val'] = True
            elif needs_test:
                # 2-way split: train/test
                plan['split_train_into_all'] = True
                plan['val_size'] = 0
        
        return plan
    
    def _load_raw_dataset(self, train: bool) -> Dataset:
        """Load raw dataset from source.
        
        Args:
            train: Whether to load training or test data
        
        Returns:
            PyTorch Dataset
        """
        if isinstance(self.data_source, (str, Path)):
            # Directory-based dataset
            data_dir = Path(self.data_source)
            subdir = 'train' if train else 'test'
            full_path = data_dir / subdir
            
            if not full_path.exists():
                raise ValueError(f"Directory not found: {full_path}")
            
            # For augmentation, we need raw PIL images (no transform)
            if self.augmentation == AugmentationStrategy.PREGENERATED:
                return datasets.ImageFolder(root=full_path, transform=None)
            else:
                transform = self.train_transform if train else self.eval_transform
                return datasets.ImageFolder(root=full_path, transform=transform)
        
        else:
            # PyTorch dataset class
            if self.root is None:
                raise ValueError("root directory is required for PyTorch datasets")
            
            # For augmentation, we need raw PIL images (no transform)
            if self.augmentation == AugmentationStrategy.PREGENERATED:
                transform = None
            else:
                transform = self.train_transform if train else self.eval_transform
            
            try:
                return self.data_source(
                    root=self.root,
                    train=train,
                    download=False,
                    transform=transform
                )
            except (RuntimeError, FileNotFoundError):
                # Download if not found
                print(f"Downloading {'train' if train else 'test'} dataset...")
                return self.data_source(
                    root=self.root,
                    train=train,
                    download=True,
                    transform=transform
                )
    
    def _create_splits(self, train_dataset: Dataset, test_dataset: Optional[Dataset] = None) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Create train/val/test splits according to plan.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset (if available)
        
        Returns:
            Tuple of (train, val, test) datasets (some may be None)
        """
        torch.manual_seed(self.seed)
        
        plan = self.split_plan
        
        if not plan['split_train_into_train_val'] and not plan['split_train_into_all']:
            # No splitting needed
            return train_dataset, None, test_dataset
        
        total_size = len(train_dataset)
        
        if plan['split_train_into_all']:
            # 3-way split
            val_size = plan['val_size']
            test_size = plan['test_size'] if plan['test_size'] > 0 else self.test_size
            
            if val_size + test_size >= total_size:
                raise ValueError(
                    f"Cannot split: val_size ({val_size}) + test_size ({test_size}) "
                    f"must be less than total size ({total_size})"
                )
            
            indices = torch.randperm(total_size).tolist()
            
            val_indices = indices[:val_size]
            test_indices = indices[val_size:val_size + test_size]
            train_indices = indices[val_size + test_size:]
            
            train_split = Subset(train_dataset, train_indices)
            val_split = Subset(train_dataset, val_indices)
            test_split = Subset(train_dataset, test_indices)
            
            return train_split, val_split, test_split
        
        elif plan['split_train_into_train_val']:
            # 2-way split: train/val
            val_size = plan['val_size']
            
            if val_size >= total_size:
                raise ValueError(
                    f"val_size ({val_size}) must be less than total size ({total_size})"
                )
            
            indices = torch.randperm(total_size).tolist()
            
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            train_split = Subset(train_dataset, train_indices)
            val_split = Subset(train_dataset, val_indices)
            
            return train_split, val_split, test_dataset
        
        return train_dataset, None, test_dataset
    
    def _apply_augmentation_on_the_fly(self, dataset: Dataset) -> Dataset:
        """Wrap dataset to apply augmentation transforms on-the-fly.
        
        Args:
            dataset: Dataset to wrap
        
        Returns:
            Wrapped dataset with augmentation
        """
        # Create augmentation transform
        if self.pil_augmentations is None:
            # Default augmentations
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            aug_transform = self.pil_augmentations
        
        # Inline AugmentedSubset implementation
        class AugmentedDataset(Dataset):
            def __init__(self, subset, augmentation):
                self.subset = subset
                self.augmentation = augmentation
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                image, label = self.subset[idx]
                if self.augmentation:
                    image = self.augmentation(image)
                return image, label
        
        return AugmentedDataset(dataset, aug_transform)
    
    def _generate_preaugmented_dataset(self, train_dataset: Dataset) -> Dataset:
        """Generate preaugmented dataset with parallel processing.
        
        Args:
            train_dataset: Training dataset (with raw PIL images)
        
        Returns:
            TensorDataset with pregenerated augmented data
        """
        output_dir = self.root / 'augmented_cache' / self.cache_key
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if not self.force_regenerate and (output_dir / 'augmented_images.pt').exists():
            print(f"Loading cached augmented data from {output_dir}")
            images = torch.load(output_dir / 'augmented_images.pt')
            labels = torch.load(output_dir / 'augmented_labels.pt')
            return TensorDataset(images, labels)
        
        # Default augmentations if not provided
        if self.pil_augmentations is None:
            pil_augs = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        else:
            pil_augs = self.pil_augmentations
        
        if self.tensor_augmentations is None:
            tensor_augs = transforms.Compose([
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            tensor_augs = self.tensor_augmentations
        
        # Convert to tensor and normalize (from train_transform)
        to_tensor = transforms.ToTensor()
        normalize = None
        for t in self.train_transform.transforms:
            if isinstance(t, transforms.Normalize):
                normalize = t
                break
        
        if normalize is None:
            # Use default normalization
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        print(f"\nGenerating augmented dataset with {self.num_workers} workers...")
        print(f"Source dataset: {len(train_dataset)} images")
        print(f"Augmentations per image: 5")
        print(f"Total augmented images: {len(train_dataset) * 5}")
        
        # Collect all images and labels
        images_labels = []
        for idx in tqdm(range(len(train_dataset)), desc="Loading images"):
            img, label = train_dataset[idx]
            images_labels.append((img, label))
        
        # Divide into chunks for parallel processing
        chunk_size = max(1, len(images_labels) // max(1, self.num_workers))
        chunks = []
        for i in range(0, len(images_labels), chunk_size):
            chunk = images_labels[i:i + chunk_size]
            chunks.append((chunk, 5, pil_augs, tensor_augs, to_tensor, normalize))
        
        # Process in parallel
        all_images = []
        all_labels = []
        
        if self.num_workers > 0:
            with Pool(processes=self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_augment_batch_worker, chunks),
                    total=len(chunks),
                    desc="Generating augmented data"
                ))
        else:
            # Serial processing
            results = []
            for chunk in tqdm(chunks, desc="Generating augmented data"):
                results.append(_augment_batch_worker(chunk))
        
        # Collect results
        for batch_images, batch_labels in results:
            all_images.extend(batch_images)
            all_labels.extend(batch_labels)
        
        # Stack into tensors
        print("Stacking tensors...")
        images_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        
        # Save to cache
        print(f"Saving to cache: {output_dir}")
        torch.save(images_tensor, output_dir / 'augmented_images.pt')
        torch.save(labels_tensor, output_dir / 'augmented_labels.pt')
        
        metadata = {
            'cache_key': self.cache_key,
            'original_size': len(train_dataset),
            'augmented_size': len(all_images),
            'n_augmentations': 5,
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return TensorDataset(images_tensor, labels_tensor)
    
    def _preload_to_device(self, dataset: Dataset, device_str: str, desc: str = "Preloading") -> TensorDataset:
        """Preload dataset to CPU or GPU memory.
        
        Args:
            dataset: Dataset to preload
            device_str: 'cpu' or 'gpu'
            desc: Description for progress bar
        
        Returns:
            TensorDataset with data on device
        """
        device = torch.device('cuda' if device_str == 'gpu' else 'cpu')
        
        print(f"{desc} to {device_str.upper()}...")
        
        images = []
        labels = []
        
        for img, label in tqdm(dataset, desc=desc):
            images.append(img)
            labels.append(label)
        
        images_tensor = torch.stack(images).to(device)
        labels_tensor = torch.tensor(labels).to(device)
        
        return TensorDataset(images_tensor, labels_tensor)
    
    def get_loaders(self) -> DataLoaders:
        """Create and return DataLoaders based on configuration.
        
        Returns:
            DataLoaders object with train/val/test loaders and metadata
        """
        # Load datasets according to plan
        train_raw = None
        test_raw = None
        
        if self.split_plan['load_train']:
            print(f"Loading train dataset...")
            train_raw = self._load_raw_dataset(train=True)
        
        if self.split_plan['load_test']:
            print(f"Loading test dataset...")
            test_raw = self._load_raw_dataset(train=False)
        
        # Create splits
        print("Creating splits...")
        train_split, val_split, test_split = self._create_splits(train_raw, test_raw)
        
        # Apply augmentation if needed
        if self.augmentation == AugmentationStrategy.PREGENERATED:
            print("\nApplying pregenerated augmentation...")
            train_split = self._generate_preaugmented_dataset(train_split)
        elif self.augmentation == AugmentationStrategy.ON_THE_FLY and train_split is not None:
            print("Setting up on-the-fly augmentation...")
            train_split = self._apply_augmentation_on_the_fly(train_split)
        
        # Preload if requested
        if self.preload is not None:
            if train_split is not None and 'train' in self.splits_needed:
                train_split = self._preload_to_device(train_split, self.preload, "Preloading train")
            if val_split is not None:
                val_split = self._preload_to_device(val_split, self.preload, "Preloading val")
            if test_split is not None and 'test' in self.splits_needed:
                test_split = self._preload_to_device(test_split, self.preload, "Preloading test")
        
        # Create DataLoaders
        num_workers = 0 if self.preload else self.num_workers
        
        train_loader = None
        val_loader = None
        test_loader = None
        
        train_size = 0
        val_size = 0
        test_size = 0
        
        if train_split is not None and 'train' in self.splits_needed:
            train_size = len(train_split)
            train_loader = DataLoader(
                train_split,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=num_workers,
                **self.loader_kwargs
            )
        
        if val_split is not None:
            val_size = len(val_split)
            val_loader = DataLoader(
                val_split,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                **self.loader_kwargs
            )
        
        if test_split is not None and 'test' in self.splits_needed:
            test_size = len(test_split)
            test_loader = DataLoader(
                test_split,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                **self.loader_kwargs
            )
        
        device_str = self.preload if self.preload else None
        
        return DataLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            batch_size=self.batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            device=device_str
        )
    
    @staticmethod
    def compute_dataset_stats(
        data_source: type,
        root: str | Path,
        num_samples: int = 5000
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute mean and std for dataset normalization.
        
        Args:
            data_source: PyTorch dataset class (e.g., datasets.CIFAR10)
            root: Root directory for data
            num_samples: Number of samples to use for computation (default: 5000)
        
        Returns:
            Tuple of (mean, std) where each is (R, G, B) tuple
        
        Example:
            >>> mean, std = DataPipeline.compute_dataset_stats(
            ...     datasets.CIFAR10,
            ...     root='./data',
            ...     num_samples=5000
            ... )
            >>> print(f"Mean: {mean}, Std: {std}")
        """
        print(f"Computing dataset statistics from {num_samples} samples...")
        
        # Load dataset with ToTensor only (no normalization)
        dataset = data_source(
            root=root,
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )
        
        # Sample random indices
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        
        # Collect samples
        images = []
        for idx in tqdm(indices, desc="Loading samples"):
            img, _ = dataset[idx]
            images.append(img)
        
        # Stack and compute stats
        images_tensor = torch.stack(images)
        mean = images_tensor.mean(dim=[0, 2, 3])
        std = images_tensor.std(dim=[0, 2, 3])
        
        mean_tuple = tuple(mean.tolist())
        std_tuple = tuple(std.tolist())
        
        print(f"Mean: {mean_tuple}")
        print(f"Std: {std_tuple}")
        
        return mean_tuple, std_tuple


def _augment_batch_worker(args):
    """Worker function for parallel augmentation.
    
    Args:
        args: Tuple of (images_labels, n_augmentations, pil_aug, tensor_aug, to_tensor, normalize)
    
    Returns:
        Tuple of (augmented_images, labels) as lists
    """
    images_labels, n_augmentations, pil_aug, tensor_aug, to_tensor, normalize = args
    
    batch_images = []
    batch_labels = []
    
    for img, label in images_labels:
        for _ in range(n_augmentations):
            # Apply PIL augmentations
            aug_img = pil_aug(img)
            
            # Convert to tensor and normalize
            tensor_img = to_tensor(aug_img)
            tensor_img = normalize(tensor_img)
            
            # Apply tensor augmentations
            tensor_img = tensor_aug(tensor_img)
            
            batch_images.append(tensor_img)
            batch_labels.append(label)
    
    return batch_images, batch_labels
