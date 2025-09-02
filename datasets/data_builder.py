import logging
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed, class_name=None):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed, class_name)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def create_dataloader_from_dataset(dataset, cfg, distributed=True):
    """Helper function to create dataloader from dataset"""
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    
    return data_loader


def build_dataloader(cfg_dataset, distributed=True, class_name=None):
    """Build train, validation, and test dataloaders with validation split support."""
    
    train_loader = None
    val_loader = None
    test_loader = None
    
    if cfg_dataset.get("train", None):
        # Build full training dataloader first
        full_train_loader = build(cfg_dataset, training=True, distributed=distributed, class_name=class_name)
        
        # Check if validation split is requested
        validation_split = cfg_dataset.get("validation_split", None)
        if validation_split and validation_split > 0:
            logger.info(f"Validation split requested: {validation_split}")
            
            # Get the underlying dataset from dataloader
            full_dataset = full_train_loader.dataset
            
            # Handle DistributedSampler case
            if hasattr(full_dataset, 'dataset'):
                # If it's a wrapped dataset (e.g., from DistributedSampler), get the original
                original_dataset = full_dataset.dataset if hasattr(full_dataset, 'dataset') else full_dataset
            else:
                original_dataset = full_dataset
            
            # Calculate split sizes
            total_size = len(original_dataset)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size
            
            logger.info(f"Splitting training data: {train_size} train, {val_size} validation samples")
            
            # Split dataset with reproducible seed
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                original_dataset, [train_size, val_size], generator=generator
            )
            
            # Create new dataloaders for train and validation
            train_loader = create_dataloader_from_dataset(train_dataset, cfg_dataset, distributed)
            val_loader = create_dataloader_from_dataset(val_dataset, cfg_dataset, distributed)
            
        else:
            # No validation split requested, use full training data
            train_loader = full_train_loader
            logger.info("No validation split requested, using full training data")
    
    # Build test loader (unchanged)
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed, class_name=class_name)

    logger.info("build dataset done")
    
    # Return all three loaders (val_loader will be None if no split)
    return train_loader, val_loader, test_loader