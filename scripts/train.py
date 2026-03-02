#!/usr/bin/env python3
"""
Complete training script for Pillar finetune with distributed training support.

This script handles:
- Config loading with OmegaConf/YAML and CLI overrides
- Distributed training with torchrun
- Model, optimizer, scheduler, and engine building
- Complete training loop with validation
- Checkpoint saving and resuming
- WandB logging

Usage:
    # Single GPU
    python scripts/train.py configs/nlst_detr_atlas.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/train.py configs/nlst_detr_atlas.yaml

    # Multi-GPU with uv and torchrun (recommended)
    uv run torchrun --nproc_per_node=8 scripts/train.py configs/nlst_detr_atlas.yaml

    # With overrides
    python scripts/train.py configs/nlst_detr_atlas.yaml --opts engine.max_epochs 100

    # With CLI shortcuts
    python scripts/train.py configs/nlst_detr_atlas.yaml --debug
    python scripts/train.py configs/nlst_detr_atlas.yaml --evaluate
    python scripts/train.py configs/nlst_detr_atlas.yaml --resume /path/to/checkpoint.ckpt
"""

import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from timm.utils import NativeScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pillar import datasets, models, engines
from pillar.utils.parsing import parse_args, dump_args
from pillar.utils.loading import get_train_dataset_loader, get_eval_dataset_loader
from pillar.utils.lr_decay import param_groups_lrd
from pillar.utils.cosine_with_warmup import CosineAnnealingWarmup, ConstantWarmup
from pillar.utils.logging import logger
from pillar.utils.misc import (
    setup_for_distributed,
    setup_dirs,
    get_model_summary,
    log_dict,
    get_is_master,
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Not using distributed mode")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()

    is_master = rank == 0
    setup_for_distributed(is_master)

    logger.info(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    return True, rank, world_size, local_rank


def setup_experiment_dirs(args):
    """Setup experiment directories for logging and checkpoints."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse experiment name from args
    if hasattr(args, "experiment") and hasattr(args.experiment, "name"):
        exp_name = args.experiment.name
    elif hasattr(args.main, "exp_name"):
        exp_name = args.main.exp_name
    else:
        exp_name = "default_experiment"

    # Setup directory structure
    log_dir = Path(args.main.log_dir) if hasattr(args.main, "log_dir") else Path("./logs")
    checkpoints_dir = args.main.checkpoints_dir if hasattr(args.main, "checkpoints_dir") else log_dir / "localization"

    if args.main.timestamp_save_dir:
        experiment_dir = Path(checkpoints_dir) / exp_name / timestamp
    else:
        experiment_dir = Path(checkpoints_dir) / exp_name
    experiment_checkpoints_dir = experiment_dir / "checkpoints"

    # Update args with directories
    args.main.experiment_dir = str(experiment_dir)
    args.main.experiment_checkpoints_dir = str(experiment_checkpoints_dir)
    args.main.timestamp = timestamp

    # Create directories (only on master)
    if get_is_master():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = experiment_dir / "config.yaml"
        dump_args(args, str(config_path), allow_overwrite=True)
        logger.info(f"Config saved to {config_path}")

    return str(experiment_dir), str(experiment_checkpoints_dir)


def init_wandb(args, is_master):
    """Initialize Weights & Biases logging."""
    if not is_master or args.main.disable_wandb:
        # Set wandb to offline/disabled mode for non-master processes
        os.environ["WANDB_MODE"] = "disabled"
        return None

    # Parse experiment configuration
    if hasattr(args, "experiment"):
        exp_name = getattr(args.experiment, "name", getattr(args.main, "exp_name", "experiment"))
        tags = getattr(args.experiment, "tags", getattr(args.main, "tags", []))
        notes = getattr(args.experiment, "notes", getattr(args.main, "notes", None))
    else:
        exp_name = getattr(args.main, "exp_name", "experiment")
        tags = getattr(args.main, "tags", [])
        notes = getattr(args.main, "notes", None)

    # Parse tags if string
    if isinstance(tags, str):
        import json

        try:
            tags = json.loads(tags)
        except:
            tags = [tags]

    wandb_config = {
        "project": args.main.wandb_project,
        "entity": args.main.wandb_entity if hasattr(args.main, "wandb_entity") else None,
        "name": exp_name,
        "config": dict(args),
        "tags": tags,
        "notes": notes,
        "dir": args.main.experiment_dir,
    }

    if hasattr(args.main, "wandb_offline") and args.main.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(**wandb_config)
    logger.info(f"WandB initialized: {run.url}")

    return run


def build_dataset(args, split):
    """Build dataset for given split."""
    logger.info(f"Building {split} dataset...")

    # Get dataset class
    dataset_type = args.dataset.type
    if dataset_type not in datasets.__dict__:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset_cls = datasets.__dict__[dataset_type]

    # Get augmentations (empty for now, handled by dataset)
    augmentations = []

    # Prepare kwargs
    shared_kwargs = dict(args.dataset.shared_dataset_kwargs) if hasattr(args.dataset, "shared_dataset_kwargs") else {}

    if split == "train":
        split_kwargs = dict(args.dataset.dataset_train_kwargs) if hasattr(args.dataset, "dataset_train_kwargs") else {}
    elif split == "dev":
        split_kwargs = dict(args.dataset.dataset_dev_kwargs) if hasattr(args.dataset, "dataset_dev_kwargs") else {}
    elif split == "test":
        split_kwargs = dict(args.dataset.dataset_test_kwargs) if hasattr(args.dataset, "dataset_test_kwargs") else {}
    else:
        split_kwargs = {}

    # Merge kwargs
    dataset_kwargs = {**shared_kwargs, **split_kwargs}

    # Create dataset
    dataset = dataset_cls(args=args, augmentations=augmentations, split_group=split, **dataset_kwargs)

    logger.info(f"{split} dataset size: {len(dataset)}")
    return dataset


def build_dataloaders(args, train_dataset, dev_dataset=None, test_dataset=None):
    """Build dataloaders for training and evaluation."""
    logger.info("Building dataloaders...")

    # Train dataloader
    train_loader = None
    if train_dataset is not None:
        train_loader = get_train_dataset_loader(args, train_dataset)
        logger.info(f"Train dataloader: {len(train_loader)} batches")

    # Dev dataloader
    dev_loader = None
    if dev_dataset is not None:
        dev_loader = get_eval_dataset_loader(
            args,
            dev_dataset,
            shuffle=False,
            multi_gpu_eval=args.dataloader.multi_gpu_eval if hasattr(args.dataloader, "multi_gpu_eval") else False,
        )
        logger.info(f"Dev dataloader: {len(dev_loader)} batches")

    # Test dataloader
    test_loader = None
    if test_dataset is not None:
        test_loader = get_eval_dataset_loader(
            args,
            test_dataset,
            shuffle=False,
            multi_gpu_eval=args.dataloader.multi_gpu_eval if hasattr(args.dataloader, "multi_gpu_eval") else False,
        )
        logger.info(f"Test dataloader: {len(test_loader)} batches")

    return train_loader, dev_loader, test_loader


def build_model(args):
    """Build model from config."""
    logger.info("Building model...")

    model_type = args.model.type
    if model_type not in models.__dict__:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cls = models.__dict__[model_type]
    model_kwargs = dict(args.model.kwargs) if hasattr(args.model, "kwargs") and args.model.kwargs else {}

    model = model_cls(args=args, **model_kwargs)

    # Log model summary
    summary = get_model_summary(model)
    log_dict(summary)

    return model


def build_optimizer(args, model):
    """Build optimizer with optional layer-wise learning rate decay."""
    logger.info("Building optimizer...")

    optimizer_type = args.optimizer.type
    optimizer_kwargs = (
        dict(args.optimizer.kwargs) if hasattr(args.optimizer, "kwargs") and args.optimizer.kwargs else {}
    )

    # Check if using layer decay
    use_layer_decay = hasattr(args.optimizer, "layer_decay") and args.optimizer.layer_decay is not None

    if use_layer_decay:
        logger.info(f"Using layer-wise learning rate decay: {args.optimizer.layer_decay}")

        # Get no weight decay parameters
        no_weight_decay_list = model.no_weight_decay() if hasattr(model, "no_weight_decay") else []

        # Build parameter groups with layer decay
        base_lr = optimizer_kwargs.get("lr", 1e-4)
        weight_decay = optimizer_kwargs.get("weight_decay", 0.05)

        param_groups = param_groups_lrd(
            model,
            weight_decay=weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=args.optimizer.layer_decay,
        )

        for group in param_groups:
            group["lr"] = base_lr * group["lr_scale"]
        # Remove lr and weight_decay from kwargs since they're in param_groups
        optimizer_kwargs_filtered = {k: v for k, v in optimizer_kwargs.items() if k not in ["lr", "weight_decay"]}
        optimizer = getattr(torch.optim, optimizer_type)(param_groups, **optimizer_kwargs_filtered)
    else:
        # Standard optimizer without layer decay
        optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), **optimizer_kwargs)

    logger.info(f"Optimizer: {optimizer_type} with {len(optimizer.param_groups)} parameter groups")

    return optimizer


def build_scheduler(args, optimizer):
    """Build learning rate scheduler."""
    if not hasattr(args.optimizer, "scheduler") or args.optimizer.scheduler is None:
        logger.info("No scheduler specified")
        return None

    scheduler_config = args.optimizer.scheduler
    scheduler_type = scheduler_config.type

    if scheduler_type is None:
        logger.info("No scheduler specified")
        return None

    logger.info(f"Building scheduler: {scheduler_type}")

    scheduler_kwargs = (
        dict(scheduler_config.kwargs) if hasattr(scheduler_config, "kwargs") and scheduler_config.kwargs else {}
    )

    # Build scheduler
    if scheduler_type == "CosineAnnealingWarmup":
        scheduler = CosineAnnealingWarmup(optimizer, **scheduler_kwargs)
    elif scheduler_type == "ConstantWarmup":
        scheduler = ConstantWarmup(optimizer, **scheduler_kwargs)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    else:
        # Try to get from torch.optim.lr_scheduler
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    logger.info(f"Scheduler: {scheduler_type}")

    return scheduler


def build_engine(args, train_dataset=None):
    """Build training engine."""
    logger.info("Building engine...")

    engine_type = args.engine.type
    if engine_type not in engines.__dict__:
        raise ValueError(f"Unknown engine type: {engine_type}")

    engine_cls = engines.__dict__[engine_type]
    engine_kwargs = dict(args.engine.kwargs) if hasattr(args.engine, "kwargs") and args.engine.kwargs else {}

    # Add max_epochs from engine config
    engine_kwargs["max_epochs"] = (
        args.engine.max_epochs if hasattr(args.engine, "max_epochs") else engine_kwargs.get("max_epochs", 50)
    )

    # Add resume path
    engine_kwargs["resume"] = (
        args.engine.kwargs.resume if hasattr(args.engine, "kwargs") and hasattr(args.engine.kwargs, "resume") else None
    )

    # Add dataset info if available
    dataset_info = train_dataset.info if train_dataset is not None and hasattr(train_dataset, "info") else None

    engine = engine_cls(args, dataset_info=dataset_info, **engine_kwargs)

    logger.info(f"Engine: {engine_type}")

    return engine


def save_checkpoint(args, epoch, model, optimizer, scheduler, engine, is_best=False):
    """Save checkpoint."""
    if not get_is_master():
        return

    ckpt_dir = args.main.experiment_checkpoints_dir

    # Prepare checkpoint state
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": engine.global_step,
        "args": args,
    }

    # Save periodic checkpoint
    if hasattr(args.main, "ckpt_freq") and epoch % args.main.ckpt_freq == 0:
        engine.save_on_master(ckpt_dir=ckpt_dir, epoch=epoch, state=state)
        logger.info(f"Saved checkpoint at epoch {epoch}")

    # Save latest checkpoint
    engine.save_on_master(ckpt_dir=ckpt_dir, epoch=-1, state=state)

    # Save best checkpoint
    if is_best:
        path = os.path.join(ckpt_dir, "best.ckpt")
        torch.save(state, path)
        logger.info(f"Saved best checkpoint at epoch {epoch}")


def load_checkpoint(args, model, optimizer, scheduler, engine):
    """Load checkpoint for resuming training."""
    resume_path = None
    if hasattr(args.engine, "kwargs") and hasattr(args.engine.kwargs, "resume"):
        resume_path = args.engine.kwargs.resume

    if resume_path is None:
        logger.info("Starting training from scratch")
        return 0

    logger.info(f"Resuming from checkpoint: {resume_path}")

    checkpoint = engine.load(resume_path, map_location="cpu")

    # Load model state
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Load engine state
    engine.global_step = checkpoint.get("global_step", 0)

    start_epoch = checkpoint["epoch"] + 1

    logger.info(f"Resumed from epoch {checkpoint['epoch']}, global_step {engine.global_step}")

    return start_epoch


def train(args):
    """Main training function."""
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Update args with distributed info
    args.main.multi_gpu = is_distributed
    args.main.global_rank = rank
    args.main.world_size = world_size
    args.main.local_rank = local_rank

    # Configure logging to only show on rank 0
    from pillar.utils.logging import set_loglevel

    set_loglevel(debug=True)

    # Set random seed
    set_seed(args.main.seed if hasattr(args.main, "seed") else None)

    # Setup experiment directories
    experiment_dir, checkpoints_dir = setup_experiment_dirs(args)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")

    # Initialize wandb
    wandb_run = init_wandb(args, get_is_master())

    # Build datasets
    train_dataset = None
    dev_dataset = None
    test_dataset = None

    force_loading_train = (
        hasattr(args.main, "force_loading_train_dataloader") and args.main.force_loading_train_dataloader
    )
    if args.main.phases.train or force_loading_train:
        train_dataset = build_dataset(args, "train")

    if hasattr(args.main.phases, "dev") and args.main.phases.dev:
        dev_dataset = build_dataset(args, "dev")

    if hasattr(args.main.phases, "test") and args.main.phases.test:
        test_dataset = build_dataset(args, "test")

    # Build dataloaders
    train_loader, dev_loader, test_loader = build_dataloaders(args, train_dataset, dev_dataset, test_dataset)

    # Build model
    model = build_model(args)

    # Move model to GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Keep reference to unwrapped model for optimizer building
    model_without_ddp = model

    # Wrap model for distributed training
    if is_distributed:
        find_unused_parameters = (
            args.main.find_unused_parameters if hasattr(args.main, "find_unused_parameters") else False
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters
        )
        logger.info("Model wrapped with DistributedDataParallel")

    # Build optimizer (use unwrapped model for param groups)
    optimizer = build_optimizer(args, model_without_ddp)

    # Build scheduler
    scheduler = build_scheduler(args, optimizer)

    # Build engine
    engine = build_engine(args, train_dataset)

    # Create loss scaler for mixed precision
    loss_scaler = NativeScaler()

    # Load checkpoint if resuming
    start_epoch = load_checkpoint(args, model, optimizer, scheduler, engine)

    # Get max epochs
    max_epochs = 50  # Default
    if hasattr(args.engine, "max_epochs"):
        max_epochs = args.engine.max_epochs
    elif hasattr(args.engine, "kwargs") and hasattr(args.engine.kwargs, "max_epochs"):
        max_epochs = args.engine.kwargs.max_epochs

    # Get training hyperparameters
    clip_grad = args.dataloader.clip_grad if hasattr(args.dataloader, "clip_grad") else None
    log_interval = engine.log_interval if hasattr(engine, "log_interval") else 50
    log_loss_components = (
        getattr(args.engine.kwargs, "log_loss_components", False) if hasattr(args.engine, "kwargs") else False
    )

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Start epoch: {start_epoch}")
    logger.info("=" * 80)

    best_metric = None
    best_epoch = -1
    monitor_metric = args.main.monitor if hasattr(args.main, "monitor") else "val_loss"
    monitor_mode = "min" if "loss" in monitor_metric else "max"

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"{'=' * 80}")
        logger.info(f"Epoch {epoch}/{max_epochs - 1}")
        logger.info(f"{'=' * 80}")

        # Set epoch for distributed sampler
        if is_distributed and args.main.phases.train and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        if args.main.phases.train:
            train_start = time.time()
            # Only pass scheduler if it's step-based
            step_scheduler = None
            if scheduler is not None and hasattr(args.optimizer.scheduler, "interval"):
                if args.optimizer.scheduler.interval == "step":
                    step_scheduler = scheduler

            engine.train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                lr_scheduler=step_scheduler,
                args=args,
                log_interval=log_interval,
                clip_grad=clip_grad,
                log_loss_components=log_loss_components,
            )
            train_time = time.time() - train_start
            logger.info(f"Training time: {train_time:.2f}s")

        # Update scheduler (epoch-based schedulers)
        if scheduler is not None and hasattr(args.optimizer.scheduler, "interval"):
            if args.optimizer.scheduler.interval == "epoch":
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric
                    pass  # Will update after validation
                else:
                    scheduler.step()

        # Validation phase
        val_metrics = None
        if hasattr(args.main.phases, "dev") and args.main.phases.dev:
            if dev_loader is not None:
                val_start = time.time()
                val_metrics = engine.evaluate(
                    model=model,
                    dataloader=dev_loader,
                    device=device,
                    epoch=epoch,
                    split="val",
                    gather_predictions=is_distributed,
                    log_loss_components=log_loss_components,
                    ckpt_dir=checkpoints_dir,
                )
                val_time = time.time() - val_start
                logger.info(f"Validation time: {val_time:.2f}s")

                # Update ReduceLROnPlateau scheduler
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None and monitor_metric in val_metrics:
                        scheduler.step(val_metrics[monitor_metric])

        # Check if this is the best model
        is_best = False
        if val_metrics is not None and monitor_metric in val_metrics:
            current_metric = val_metrics[monitor_metric]
            if isinstance(current_metric, torch.Tensor):
                current_metric = current_metric.item()

            if best_metric is None:
                is_best = True
            elif monitor_mode == "min":
                is_best = current_metric < best_metric
            else:
                is_best = current_metric > best_metric

            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                logger.info(f"New best {monitor_metric}: {best_metric:.4f} at epoch {epoch}")

        # Save checkpoint
        save_checkpoint(args, epoch, model, optimizer, scheduler, engine, is_best=is_best)

    # Test phase
    if args.main.phases.test:
        if test_loader is not None:
            logger.info("\n" + "=" * 80)
            logger.info("Running test evaluation")
            logger.info("=" * 80)

            test_start = time.time()
            test_metrics = engine.evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                epoch=start_epoch,
                split="test",
                gather_predictions=is_distributed,
                log_loss_components=log_loss_components,
                ckpt_dir=checkpoints_dir,
            )
            test_time = time.time() - test_start
            logger.info(f"Test time: {test_time:.2f}s")

    # Cleanup
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    if best_metric is not None:
        logger.info(f"Best {monitor_metric}: {best_metric:.4f} at epoch {best_epoch}")
    logger.info("\n" + "=" * 80)

    if wandb_run is not None:
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Run training
    train(args)


if __name__ == "__main__":
    main()
