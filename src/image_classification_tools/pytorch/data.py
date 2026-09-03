'''Data loading and preprocessing for image classification datasets.

This module provides a simplified, unified API for loading datasets, splitting them,
and creating PyTorch DataLoaders with support for GPU/CPU preloading and data augmentation.
'''

import json
import shutil
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


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

    
    def memory_estimate(self) -> float:
        """Estimate memory usage for preloaded data.
        
        Returns:
            Estimated memory use in GB as float"
            Returns None if data not preloaded
        """

        if self.device is None:
            return None
        
        # Estimate: assume 32x32x3 RGB images with float32 (4 bytes per value)
        bytes_per_image = 32 * 32 * 3 * 4
        total_samples = self.train_size + self.val_size + self.test_size
        total_bytes = total_samples * bytes_per_image
        total_gb = total_bytes / (1024 ** 3)
        
        return total_gb


class DataPipeline:
    """Unified data loading pipeline with auto-detection and intelligent splitting.
    
    Data augmentation is decoupled from training: augmented datasets are pregenerated 
    and saved to disk, then loaded during training. The preload parameter controls 
    whether data is loaded lazily from disk, preloaded to CPU, or preloaded to GPU.
    
    Examples:
        Basic usage without augmentation, GPU preloading:

        >>> loaders = DataPipeline(
        ...     data_source=datasets.CIFAR10,
        ...     data_dir='./data/pytorch/cifar10',
        ...     split='train/val/test',
        ...     train_transform=my_transform,
        ...     eval_transform=my_transform,
        ...     preload='gpu'
        ... ).get_loaders()
        >>> train_loader = loaders.train

        With augmentation (pregenerated and saved to disk):

        >>> loaders = DataPipeline(
        ...     data_source=datasets.CIFAR10,
        ...     data_dir='./data/pytorch/cifar10',
        ...     split='train/val/test',
        ...     train_transform=eval_transform,
        ...     eval_transform=eval_transform,
        ...     preload='cpu',
        ...     n_augmentations=5,
        ...     augmented_dataset_name='strong_aug_v1',  # Saved to ./data/pytorch/augmented_cifar10/strong_aug_v1/
        ...     pil_augmentations=my_pil_augs
        ... ).get_loaders()
    """
    
    def __init__(
        self,
        data_source: type | str | Path,
        data_dir: Optional[str | Path] = None,
        split: str = 'train/val/test',
        val_size: int = 10000,
        test_size: int = 10000,
        batch_size: int = 128,
        num_workers: int = 0,
        train_transform: Optional[transforms.Compose] = None,
        eval_transform: Optional[transforms.Compose] = None,
        shuffle_train: bool = True,
        preload: Optional[str] = None,
        n_augmentations: int = 5,
        augmented_dataset_name: Optional[str] = None,
        pil_augmentations: Optional[transforms.Compose] = None,
        tensor_augmentations: Optional[transforms.Compose] = None,
        force_regenerate: bool = False,
        seed: int = 42,
        **loader_kwargs
    ):
        """Initialize DataPipeline.
        
        Args:
            data_source: PyTorch dataset class (e.g., datasets.CIFAR10) or path to data directory
            split: Desired split outcome. Must be one of: 'train', 'train/val', 'train/test', 'train/val/test'
            data_dir: Root directory for dataset storage (required for PyTorch datasets)
            train_transform: Transform for training data (optional, defaults to eval_transform if not provided)
            eval_transform: Transform for validation/test data (optional, defaults to train_transform if not provided)
            preload: Loading strategy: 'gpu', 'cpu', or None for lazy loading from disk
            pil_augmentations: PIL augmentation transforms (flip, rotate, etc.). If provided, augmented data will be pregenerated.
            tensor_augmentations: Tensor augmentation transforms (blur, erasing, etc.). Applied during pregeneration.
            n_augmentations: Number of augmented copies per image (default: 5)
            augmented_dataset_name: Name for augmented dataset directory. Defaults to 'depth_{n_augmentations}'. 
                                   Saved to {data_dir.parent}/augmented_{data_dir.name}/{augmented_dataset_name}/
            val_size: Number of validation samples (default: 10000)
            test_size: Number of test samples for 3-way splits (default: 10000)
            batch_size: Batch size for all loaders (default: 128)
            num_workers: Number of workers for DataLoader and augmentation (default: 0)
            seed: Random seed for reproducible splits (default: 42)
            shuffle_train: Whether to shuffle training data (default: True)
            force_regenerate: Force regeneration of cached augmented data (default: False)
            **loader_kwargs: Additional arguments passed to DataLoader
        
        Raises:
            ValueError: If split format invalid or incompatible config
        """

        self.data_source = data_source
        self.split_str = split
        self.data_dir = Path(data_dir) if data_dir else None
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.preload = preload
        self.pil_augmentations = pil_augmentations
        self.tensor_augmentations = tensor_augmentations
        self.n_augmentations = n_augmentations
        
        # Auto-generate augmented dataset name if augmentations are specified
        if augmented_dataset_name is None and (pil_augmentations is not None or tensor_augmentations is not None):
            self.augmented_dataset_name = f"depth_{n_augmentations}"
        else:
            self.augmented_dataset_name = augmented_dataset_name
        
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
        """Validate configuration."""
        
        # Auto-set transforms if only one is provided
        if self.train_transform is None and self.eval_transform is not None:
            self.train_transform = self.eval_transform
        elif self.eval_transform is None and self.train_transform is not None:
            self.eval_transform = self.train_transform
        elif self.train_transform is None and self.eval_transform is None:
            raise ValueError(
                "At least one of train_transform or eval_transform must be provided."
            )
        
        # Validate preload
        if self.preload is not None:
            if self.preload not in ['gpu', 'cpu'] and not self.preload.startswith('cuda:'):
                raise ValueError(
                    f"preload must be 'gpu', 'cpu', 'cuda:N', or None, got: {self.preload}"
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
            
            # For augmentation pregeneration, we need raw PIL images (no transform)
            if self.pil_augmentations is not None or self.tensor_augmentations is not None:
                return datasets.ImageFolder(root=full_path, transform=None)
            else:
                transform = self.train_transform if train else self.eval_transform
                return datasets.ImageFolder(root=full_path, transform=transform)
        
        else:
            # PyTorch dataset class
            if self.data_dir is None:
                raise ValueError("data_dir is required for PyTorch datasets")
            
            # For augmentation pregeneration, we need raw PIL images (no transform)
            if self.pil_augmentations is not None or self.tensor_augmentations is not None:
                transform = None
            else:
                transform = self.train_transform if train else self.eval_transform
            
            try:
                return self.data_source(
                    root=self.data_dir,
                    train=train,
                    download=False,
                    transform=transform
                )

            except (RuntimeError, FileNotFoundError):

                # Download if not found
                print(f"Downloading {'train' if train else 'test'} dataset...")
                return self.data_source(
                    root=self.data_dir,
                    train=train,
                    download=True,
                    transform=transform
                )


    def _create_splits(
            self,
            train_dataset: Dataset,
            test_dataset: Optional[Dataset] = None
    ) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
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


    def _apply_transform_to_dataset(self, dataset: Dataset, transform: transforms.Compose) -> Dataset:
        """Wrap dataset to apply a transform.
        
        Args:
            dataset: Dataset to wrap
            transform: Transform to apply
        
        Returns:
            Wrapped dataset with transform
        """
        class TransformedDataset(Dataset):
            def __init__(self, base_dataset, transform):

                self.dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):

                image, label = self.dataset[idx]

                if self.transform:
                    image = self.transform(image)

                return image, label
        
        return TransformedDataset(dataset, transform)


    def _generate_preaugmented_dataset(self, train_dataset: Dataset) -> Dataset:
        """Generate preaugmented dataset with parallel processing.
        
        Args:
            train_dataset: Training dataset (with raw PIL images)
        
        Returns:
            TensorDataset with pregenerated augmented data
        """
        # Construct output directory: {data_dir.parent}/augmented_{data_dir.name}/{augmented_dataset_name}/
        augmented_base_dir = self.data_dir.parent / f"augmented_{self.data_dir.name}"
        output_dir = augmented_base_dir / self.augmented_dataset_name
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
        print(f"Augmentations per image: {self.n_augmentations}")
        print(f"Total augmented images: {len(train_dataset) * self.n_augmentations}")
        
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
            chunks.append((chunk, self.n_augmentations, pil_augs, tensor_augs, to_tensor, normalize))
        
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
        
        # Calculate memory size estimate
        bytes_total = images_tensor.numel() * images_tensor.element_size() + labels_tensor.numel() * labels_tensor.element_size()
        memory_size_gb = bytes_total / (1024 ** 3)
        
        # Get image shape and number of classes
        sample_image = all_images[0]
        image_shape = list(sample_image.shape)
        num_classes = len(set(all_labels))
        
        # Save to cache
        print(f"Saving to cache: {output_dir}")
        torch.save(images_tensor, output_dir / 'augmented_images.pt')
        torch.save(labels_tensor, output_dir / 'augmented_labels.pt')
        
        # Enhanced metadata with more details
        import sys
        from datetime import datetime
        
        # Get string representations of augmentations
        pil_aug_strs = [str(t) for t in pil_augs.transforms] if hasattr(pil_augs, 'transforms') else [str(pil_augs)]
        tensor_aug_strs = [str(t) for t in tensor_augs.transforms] if hasattr(tensor_augs, 'transforms') else [str(tensor_augs)]
        
        metadata = {
            'dataset_name': self.augmented_dataset_name,
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_dataset': self.data_source.__name__ if hasattr(self.data_source, '__name__') else str(self.data_source),
            'original_size': len(train_dataset),
            'n_augmentations': self.n_augmentations,
            'total_images': len(all_images),
            'memory_size_gb': round(memory_size_gb, 3),
            'image_shape': image_shape,
            'num_classes': num_classes,
            'split_sizes': {
                'train': None,  # Will be updated later if needed
                'val': None,
                'test': None
            },
            'pil_augmentations': pil_aug_strs,
            'tensor_augmentations': tensor_aug_strs,
            'pytorch_version': torch.__version__,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return TensorDataset(images_tensor, labels_tensor)


    def _preload_to_device(
            self,
            dataset: Dataset,
            device_str: str,
            desc: str = "Preloading"
    ) -> TensorDataset:
        """Preload dataset to CPU or GPU memory.
        
        Args:
            dataset: Dataset to preload
            device_str: 'cpu', 'gpu', or 'cuda:N' (specific GPU)
            desc: Description for progress bar
        
        Returns:
            TensorDataset with data on device
            
        Raises:
            RuntimeError: If dataset doesn't fit in device memory
        """
        # Handle device string: 'cpu', 'gpu' (default cuda:0), or 'cuda:N'
        if device_str == 'gpu':
            device = torch.device('cuda:0')  # Default to GPU 0
            device_display = 'GPU:0'

        elif device_str.startswith('cuda:'):
            device = torch.device(device_str)
            device_display = device_str.upper()

        else:
            device = torch.device('cpu')
            device_display = 'CPU'
        
        print(f"{desc} to {device_display}...")
        
        images = []
        labels = []
        
        try:
            for img, label in tqdm(dataset, desc=desc):
                images.append(img)
                labels.append(label)
            
            images_tensor = torch.stack(images).to(device)
            labels_tensor = torch.tensor(labels).to(device)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Calculate approximate memory requirement
                if images:
                    sample_img = images[0]
                    bytes_per_image = sample_img.numel() * sample_img.element_size()
                    total_bytes = len(dataset) * bytes_per_image
                    total_gb = total_bytes / (1024 ** 3)
                    
                    raise RuntimeError(
                        f"\n{'='*70}\n"
                        f"OUT OF MEMORY ERROR\n"
                        f"{'='*70}\n"
                        f"Failed to preload dataset to {device_display}.\n"
                        f"Dataset size: {len(dataset):,} samples\n"
                        f"Estimated memory required: {total_gb:.2f} GB\n"
                        f"\nSuggestions:\n"
                        f"  1. Use preload=None for lazy loading from disk\n"
                        f"  2. Use preload='cpu' instead of 'gpu' (if currently using GPU)\n"
                        f"  3. Reduce batch_size or dataset size\n"
                        f"  4. Use fewer augmentations (reduce n_augmentations)\n"
                        f"  5. Use on-the-fly augmentation instead of pregenerated\n"
                        f"{'='*70}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"Out of memory when preloading to {device_display}. "
                        f"Try using preload=None for lazy loading from disk."
                    ) from e
            else:
                raise
                
        except MemoryError as e:
            # CPU memory error
            if images:
                sample_img = images[0]
                bytes_per_image = sample_img.numel() * sample_img.element_size()
                total_bytes = len(dataset) * bytes_per_image
                total_gb = total_bytes / (1024 ** 3)
                
                raise MemoryError(
                    f"\n{'='*70}\n"
                    f"OUT OF MEMORY ERROR\n"
                    f"{'='*70}\n"
                    f"Failed to preload dataset to CPU memory.\n"
                    f"Dataset size: {len(dataset):,} samples\n"
                    f"Estimated memory required: {total_gb:.2f} GB\n"
                    f"\nSuggestions:\n"
                    f"  1. Use preload=None for lazy loading from disk\n"
                    f"  2. Reduce batch_size or dataset size\n"
                    f"  3. Use fewer augmentations (reduce n_augmentations)\n"
                    f"  4. Use on-the-fly augmentation instead of pregenerated\n"
                    f"  5. Close other applications to free up RAM\n"
                    f"{'='*70}"
                ) from e
            else:
                raise MemoryError(
                    "Out of memory when preloading to CPU. "
                    "Try using preload=None for lazy loading from disk."
                ) from e
        
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
        
        # Apply augmentation if specified
        if self.pil_augmentations is not None or self.tensor_augmentations is not None:
            print("\nPregenerating augmented dataset...")
            train_split = self._generate_preaugmented_dataset(train_split)
            
            # Val/test splits need eval_transform applied
            # since they were loaded without transforms (as Subsets of raw dataset)
            if val_split is not None:
                val_split = self._apply_transform_to_dataset(val_split, self.eval_transform)
            if test_split is not None:
                test_split = self._apply_transform_to_dataset(test_split, self.eval_transform)
        
        # Preload if requested (skip test set to save VRAM)
        if self.preload is not None:
            if train_split is not None and 'train' in self.splits_needed:
                train_split = self._preload_to_device(train_split, self.preload, "Preloading train")
            if val_split is not None:
                val_split = self._preload_to_device(val_split, self.preload, "Preloading val")
            # Test set is NOT preloaded to save VRAM for training/validation
        
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
        data_dir: str | Path,
        num_samples: int = 5000
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute mean and std for dataset normalization.
        
        Args:
            data_source: PyTorch dataset class (e.g., datasets.CIFAR10)
            data_dir: Root directory for dataset storage
            num_samples: Number of samples to use for computation (default: 5000)
        
        Returns:
            Tuple of (mean, std) where each is (R, G, B) tuple
        
        Example:
            >>> mean, std = DataPipeline.compute_dataset_stats(
            ...     datasets.CIFAR10,
            ...     data_dir='./data/pytorch/cifar10',
            ...     num_samples=5000
            ... )
            >>> print(f"Mean: {mean}, Std: {std}")
        """
        print(f"Computing dataset statistics from {num_samples} samples...")
        
        # Load dataset with ToTensor only (no normalization)
        dataset = data_source(
            root=data_dir,
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
