import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import hydra
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.models.discriminator import MetricGAN_Discriminator
from src.data import VoiceBankDataset, StepSampler
from src.solver import Solver
from src.utils import load_model

torch.backends.cudnn.benchmark = True


def setup_logger(name):
    """Set up logger"""
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)

def run(args):
        
    # Create and initialize logger
    logger = setup_logger("train")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare model params (skip pretrained loading if resuming from checkpoint)
    model_params = OmegaConf.to_container(args.model, resolve=True)
    if args.continue_from is not None and 'load_pretrained_weights' in model_params:
        model_params['load_pretrained_weights'] = False
        logger.info("[Resume] Skipping pretrained weights loading (will load from checkpoint)")

    # Load model
    model = load_model(model_params, device)

    # Calculate and log the total number of parameters and model size
    logger.info(f"Selected model: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1_000_000
    logger.info(f"Model's parameters: {total_params_m:.2f} M")

    if args.optim == "adam":
        optim_class = torch.optim.Adam
    elif args.optim == "adamW" or args.optim == "adamw":
        optim_class = torch.optim.AdamW

    generator_only = bool(args.get("generator_only", False))
    discriminator = (
        None if generator_only else MetricGAN_Discriminator().to(device)
    )

    # optimizer
    optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)

    optim_disc = (
        None
        if discriminator is None
        else optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)
    )
    
    # scheduler
    scheduler = None
    scheduler_disc = None

    if args.lr_decay is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay, last_epoch=-1)
        if optim_disc is not None:
            scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(
                optim_disc, gamma=args.lr_decay, last_epoch=-1
            )

    sfi_config = args.model.get("sfi", {})
    sfi_enabled = bool(sfi_config.get("enabled", False))
    reference_sample_rate = int(
        sfi_config.get("reference_sample_rate", args.sampling_rate)
    )

    # Determine training segment size.  SFI uses physical seconds so the same
    # duration is preserved after selecting a target sampling rate.
    from src.receptive_field import compute_receptive_field, rf_to_segment
    raw_segment = args.dset.get("segment", 64000)
    segment_seconds = args.dset.get("segment_seconds", None) if sfi_enabled else None
    if segment_seconds is not None:
        segment_seconds = float(segment_seconds)
        segment = round(segment_seconds * reference_sample_rate)
        logger.info(
            "SFI segment: %.3f s -> %d samples at reference rate %d Hz",
            segment_seconds,
            segment,
            reference_sample_rate,
        )
    elif raw_segment == "auto":
        segment = rf_to_segment(args.model, sampling_rate=args.sampling_rate)
        rf = compute_receptive_field(args.model, sampling_rate=args.sampling_rate)
        logger.info(f"Auto segment from RF: {rf.total_rf_frames} frames = "
                    f"{rf.total_rf_samples} samples -> aligned to {segment} samples "
                    f"({segment / args.sampling_rate * 1000:.1f} ms)")
    else:
        segment = int(raw_segment)

    # Store resolved segment for solver
    OmegaConf.update(args, "segment", segment, force_add=True)

    # Load VoiceBank+DEMAND dataset
    hf_dataset_id = args.dset.get("hf_dataset_id", "JacobLinCool/VoiceBank-DEMAND-16k")
    logger.info(f"Loading VoiceBank dataset from HuggingFace: {hf_dataset_id}")

    _vb_dataset = load_dataset(hf_dataset_id)
    trainset = _vb_dataset["train"]
    testset = _vb_dataset["test"]

    tr_dataset = VoiceBankDataset(
        trainset,
        segment=None if segment_seconds is not None else segment,
        segment_seconds=segment_seconds,
        sampling_rate=reference_sample_rate,
        return_sample_rate=sfi_enabled,
    )
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    va_dataset = VoiceBankDataset(
        testset,
        segment=None,
        sampling_rate=reference_sample_rate,
        return_sample_rate=sfi_enabled,
    )
    va_loader = DataLoader(
        dataset=va_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    ev_dataset = VoiceBankDataset(testset, segment=None, with_id=True, with_text=True)
    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )
    tt_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        sampler=StepSampler(len(ev_dataset), 100),
        num_workers=args.num_workers,
        pin_memory=True
    )

    ev_loader_list = {"mixed": ev_loader}
    tt_loader_list = {"mixed": tt_loader}
    logger.info(f"Train samples: {len(tr_dataset)}, Valid/Test samples: {len(va_dataset)}")

    dataloader = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader_list": ev_loader_list,
        "tt_loader_list": tt_loader_list,
    }

    # Solver
    solver = Solver(
        data=dataloader,
        model=model,
        discriminator=discriminator,
        optim=optim,
        optim_disc=optim_disc,
        scheduler=scheduler,
        scheduler_disc=scheduler_disc,
        args=args,
        logger=logger,
        device=device
    )
    solver.train()
    sys.exit(0)

def _main(args):
    logger = setup_logger("main")
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    run(args)

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(args):
    logger = setup_logger("main")
    try:
        _main(args)
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error occurred in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
