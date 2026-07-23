import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import logging
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.compute_metrics import compute_metrics
from src.stft import mag_pha_stft, mag_pha_istft
from src.utils import bold, LogProgress


def evaluate(
    args: DictConfig,
    model: torch.nn.Module,
    data_loader_list: Dict[str, DataLoader],
    logger: logging.Logger,
    epoch: Optional[int] = None,
    stft_args: Optional[Dict[str, Any]] = None,
    latency_id: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Model evaluation

    Args:
        args: Evaluation settings (OmegaConf)
        model: Model to evaluate
        data_loader_list: Dictionary of dataloaders by SNR {"0": loader, "5": loader, ...}
        logger: Logger instance
        epoch: Current epoch (during training)
        stft_args: STFT parameters

    Returns:
        Dictionary of metrics by SNR
        {
            "0dB": {"pesq": 2.5, "stoi": 0.85, ...},
            "5dB": {"pesq": 2.8, "stoi": 0.90, ...}
        }
    """
    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""

    metrics = {}
    model.eval()

    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        results = []
        with torch.no_grad():
            for data in iterator:
                noisy, clean, _, _ = data

                noisy_com = mag_pha_stft(noisy, **stft_args)[2].to(args.device)
                if latency_id is not None:
                    clean_mag_hat, clean_pha_hat, _ = model(noisy_com, latency_id=latency_id)
                else:
                    clean_mag_hat, clean_pha_hat, _ = model(noisy_com)

                clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                clean = clean.squeeze().detach().cpu().numpy()
                clean_hat = clean_hat.squeeze().detach().cpu().numpy()
                if len(clean) != len(clean_hat):
                    length = min(len(clean), len(clean_hat))
                    clean = clean[0:length]
                    clean_hat = clean_hat[0:length]

                results.append(compute_metrics(clean, clean_hat))

        pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
        metrics[f'{snr}dB'] = {
            "pesq": pesq,
            "stoi": stoi,
            "csig": csig,
            "cbak": cbak,
            "covl": covl,
            "segSNR": segSNR
        }
        logger.info(bold(f"{prefix}Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))

    return metrics



if __name__=="__main__":
    import argparse
    from src.data import VoiceBankDataset
    from src.utils import load_model, load_checkpoint, get_stft_args_from_config
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory.")
    parser.add_argument("--chkpt_file", type=str, required=True, help="Checkpoint file name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers.")
    parser.add_argument("--log_file", type=str, default="output.log", help="Log file name.")
    parser.add_argument("--latency_id", type=str, default=None,
                        help="Latency id for latency-expert checkpoints.")

    args = parser.parse_args()
    device = args.device

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    conf = OmegaConf.load(args.model_config)
    conf.device = device

    # Load model and checkpoint
    model = load_model(conf.model, device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, device)

    # Prepare STFT args from model config
    stft_args = get_stft_args_from_config(conf.model)

    # Load VoiceBank+DEMAND test set
    hf_dataset_id = conf.dset.get("hf_dataset_id", "JacobLinCool/VoiceBank-DEMAND-16k")
    testset = load_dataset(hf_dataset_id, split="test")

    ev_dataset = VoiceBankDataset(testset, segment=None, with_id=True, with_text=True)
    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )
    ev_loader_list = {"mixed": ev_loader}

    logger.info(f"Dataset: VoiceBank-DEMAND")
    logger.info(f"Model: Backbone")
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    if args.latency_id:
        logger.info(f"Latency id: {args.latency_id}")
    logger.info(f"Device: {device}")

    evaluate(args=conf,
            model=model,
            data_loader_list=ev_loader_list,
            logger=logger,
            epoch=None,
            stft_args=stft_args,
            latency_id=args.latency_id)
