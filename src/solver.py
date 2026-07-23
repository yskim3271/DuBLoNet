import os
import time
import shutil
import random
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from .stft import mag_pha_istft, mag_pha_stft
from .utils import copy_state, batch_pesq, anti_wrapping_function, \
                  phase_losses, get_stft_args_from_config
from .v2_contract import (
    BalancedOperatingPointSchedule,
    LATENCY_FUTURE_FRAMES,
    TARGET_SAMPLE_RATES,
)


class Solver(object):
    def __init__(
        self,
        data,
        model,
        discriminator,
        optim,
        optim_disc,
        scheduler,
        scheduler_disc,
        args,
        logger,
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.tt_loader_list = data['tt_loader_list']      # Test Time Evaluation DataLoader

        self.model = model
        self.discriminator = discriminator
        self.optim = optim
        self.optim_disc = optim_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc

        # loss weights
        self.loss = args.loss

        # logger
        self.logger = logger

        # dataset & STFT
        self.segment = args.segment
        self.stft_args = get_stft_args_from_config(args.model)
        # Basic config
        self.device = device or torch.device(args.device)

        self.max_steps = args.max_steps

        self.validation_interval = args.validation_interval
        self.summary_interval = args.summary_interval
        self.log_interval = args.log_interval
        self.best_models_num = args.best_models_num
        self.scheduler_step_interval = getattr(args, 'scheduler_step_interval', None)

        # Checkpoint settings
        self.continue_from = args.continue_from

        self.writer = None
        self.best_models = []
        self.log_dir = args.log_dir
        self.num_workers = args.num_workers
        self.args = args

        sfi_config = args.model.get("sfi", {})
        self.sfi_enabled = bool(sfi_config.get("enabled", False))
        self.reference_sample_rate = int(
            sfi_config.get("reference_sample_rate", args.sampling_rate)
        )
        self.validation_sample_rate = int(
            args.get("validation_sample_rate", self.reference_sample_rate)
        )
        self.generator_only = bool(args.get("generator_only", False))
        self.operating_schedule = None
        if not self.generator_only and (
            self.discriminator is None or self.optim_disc is None
        ):
            raise ValueError(
                "MetricGAN training requires discriminator and optimizer_disc"
            )

        self.step_start = 0
        self.is_latency_expert = hasattr(self.model, "get_latency_ids")
        self.latency_ids = self.model.get_latency_ids() if self.is_latency_expert else []
        self.latency_sampling = "uniform"
        if self.is_latency_expert:
            latency_expert_cfg = getattr(args.model, "latency_expert", {})
            if hasattr(latency_expert_cfg, "get"):
                self.latency_sampling = latency_expert_cfg.get("latency_sampling", "uniform")
            if self.latency_sampling not in {"uniform", "balanced_grid"}:
                raise ValueError(
                    f"Unsupported latency_sampling: {self.latency_sampling}. "
                    "Expected 'uniform' or 'balanced_grid'."
                )
            self.logger.info(
                "Latency expert training enabled | ids=%s | sampling=%s",
                self.latency_ids,
                self.latency_sampling,
            )

        if self.sfi_enabled:
            if not self.is_latency_expert:
                raise ValueError("v2 SFI training requires a latency-expert model")
            latency_config = args.model.get("latency_expert", {})
            future_frames = latency_config.get(
                "future_frames", LATENCY_FUTURE_FRAMES
            )
            target_sample_rates = args.get(
                "target_sample_rates",
                sfi_config.get("target_sample_rates", TARGET_SAMPLE_RATES),
            )
            if self.latency_sampling != "balanced_grid":
                raise ValueError(
                    "v2 SFI training requires latency_sampling='balanced_grid'"
                )
            self.operating_schedule = BalancedOperatingPointSchedule(
                sample_rates=list(target_sample_rates),
                future_frames=dict(future_frames),
                seed=int(args.seed),
            )
            self.logger.info(
                "SFI operating grid enabled | rates=%s | latency=%s | "
                "cycle=%d | generator_only=%s",
                list(self.operating_schedule.sample_rates),
                self.operating_schedule.future_frames,
                self.operating_schedule.cycle_size,
                self.generator_only,
            )

        # Initialize or resume (checkpoint loading)
        self._reset()

    def _save_model_checkpoint(self, steps, state_dict):
        """ Save model checkpoint. """
        package_model = {}
        package_model['model'] = copy_state(state_dict)

        tmp_path = "model.tmp"
        save_path = f"model_{steps}.th"
        torch.save(package_model, tmp_path)
        os.rename(tmp_path, save_path)

    def _save_states_checkpoint(self, step):
        """ Save states checkpoint. """
        package = {}
        package['model'] = copy_state(self.model.state_dict())
        package['best_models'] = self.best_models
        if self.discriminator is not None:
            package['discriminator'] = copy_state(self.discriminator.state_dict())
        package['optimizer'] = self.optim.state_dict()
        if self.optim_disc is not None:
            package['optimizer_disc'] = self.optim_disc.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        package['args'] = self.args
        package['step'] = step
        if self.operating_schedule is not None:
            package['operating_schedule'] = self.operating_schedule.state_dict()
        # Write to a temporary file first
        tmp_path = "states.tmp"
        save_path = "states.th"
        torch.save(package, tmp_path)
        os.rename(tmp_path, save_path)

    def _update_best_models(self, steps, valid_pesq_value):
        """Maintain top-k models by validation PESQ. """
        entry = {
            "steps": steps,
            "valid_pesq_value": valid_pesq_value,
            "model": copy_state(self.model.state_dict())
        }
        self.best_models.append(entry)
        # Keep only top k by PESQ in descending order
        self.best_models.sort(key=lambda x: x["valid_pesq_value"], reverse=True)
        if len(self.best_models) > self.best_models_num:
            for evicted in self.best_models[self.best_models_num:]:
                old_path = f"model_{evicted['steps']}.th"
                if os.path.exists(old_path):
                    os.remove(old_path)
            self.best_models = self.best_models[:self.best_models_num]

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            self.logger.info(f'Loading checkpoint model: {self.continue_from}')
            if not os.path.exists(self.continue_from):
                raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")

            # Attempt to copy the 'tensorbd' directory (TensorBoard logs) if it exists
            src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
            dst_tb_dir = self.log_dir

            if os.path.exists(src_tb_dir):
                if not os.path.exists(dst_tb_dir):
                    shutil.copytree(src_tb_dir, dst_tb_dir)
                else:
                    self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
            self.writer = SummaryWriter(log_dir=dst_tb_dir)

            # loads the checkpoint file from disk
            ckpt_path = os.path.join(self.continue_from, 'states.th')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            package = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            model_state = package['model']
            model_disc_state = package.get('discriminator', None)
            optim_state = package.get('optimizer', None)
            optim_disc_state = package.get('optimizer_disc', None)
            scheduler_state = package.get('scheduler', None)
            scheduler_disc_state = package.get('scheduler_disc', None)
            self.best_models = package.get('best_models', [])
            self.step_start = package.get('step', 0) if optim_state is not None else 0

            schedule_state = package.get('operating_schedule')
            if schedule_state is not None:
                if self.operating_schedule is None:
                    raise ValueError(
                        "Checkpoint contains an operating schedule but the current config does not"
                    )
                self.operating_schedule.load_state_dict(schedule_state)

            self.model.load_state_dict(model_state)
            if optim_state is not None:
                self.optim.load_state_dict(optim_state)
            else:
                self.logger.warning(
                    "Checkpoint has no optimizer state; loaded model weights only "
                    "and will start optimizer/scheduler from scratch."
                )

            if model_disc_state is not None and self.discriminator is not None:
                self.discriminator.load_state_dict(model_disc_state)
            if optim_disc_state is not None and self.optim_disc is not None:
                self.optim_disc.load_state_dict(optim_disc_state)

            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            if self.scheduler_disc is not None and scheduler_disc_state is not None:
                self.scheduler_disc.load_state_dict(scheduler_disc_state)

        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)


    def _infinite_loader(self):
        """Yield batches from tr_loader infinitely."""
        while True:
            for data in self.tr_loader:
                yield data

    def train(self):
        self.logger.info("Training for %d steps", self.max_steps)

        if self.step_start != 0:
            self.logger.info("Resuming training from step %d", self.step_start + 1)

        loader_iter = self._infinite_loader()

        for step in range(self.step_start + 1, self.max_steps + 1):
            start = time.time()

            data = next(loader_iter)
            noisy, clean = data[0], data[1]
            source_sample_rate = data[2] if self.sfi_enabled else None
            operating_point = (
                self.operating_schedule.next()
                if self.operating_schedule is not None
                else None
            )
            loss_dict = self._run_one_step(
                noisy,
                clean,
                source_sample_rate=source_sample_rate,
                operating_point=operating_point,
            )

            if step % self.log_interval == 0:
                lr = self.optim.param_groups[0]['lr']
                info = " | ".join(f"{k} {v:.4f}" for k, v in loss_dict.items())
                self.logger.info(f"Train | Step {step}/{self.max_steps} | LR {lr:.6f} | {1/(time.time() - start):.1f} iters/s | {info}")

            if step % self.summary_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"Train/{key}_Loss", value, step)

            if step % self.validation_interval == 0:
                val_pesq_score = self._run_validation(step)
                self._update_best_models(step, val_pesq_score)
                if any(m['steps'] == step for m in self.best_models):
                    self._save_model_checkpoint(step, self.model.state_dict())
                self._save_states_checkpoint(step)

            if self.scheduler_step_interval and step % self.scheduler_step_interval == 0:
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.scheduler_disc is not None:
                    self.scheduler_disc.step()

        self.logger.info("-" * 70)
        self.logger.info("Training Completed")
        if self.best_models:
            best = self.best_models[0]
            self.logger.info(f"Best Model | Steps: {best['steps']}, Valid PESQ: {best['valid_pesq_value']:.4f}")
        self.logger.info("-" * 70)
        self.writer.close()

    @staticmethod
    def _batch_sample_rate(value):
        """Return one integer rate and reject mixed-rate tensor batches."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            rates = {int(item) for item in value.detach().cpu().reshape(-1).tolist()}
        elif isinstance(value, (list, tuple)):
            rates = {int(item) for item in value}
        else:
            rates = {int(value)}
        if len(rates) != 1:
            raise ValueError(f"A training batch must contain one sample rate: {rates}")
        return rates.pop()

    @staticmethod
    def _resample_pair(noisy, clean, source_sample_rate, target_sample_rate):
        if source_sample_rate == target_sample_rate:
            return noisy, clean
        from torchaudio.functional import resample

        noisy = resample(noisy, source_sample_rate, target_sample_rate)
        clean = resample(clean, source_sample_rate, target_sample_rate)
        common_length = min(noisy.shape[-1], clean.shape[-1])
        return noisy[..., :common_length], clean[..., :common_length]

    def _run_one_step(
        self,
        noisy,
        clean,
        source_sample_rate=None,
        operating_point=None,
    ):

        self.model.train()
        if not self.generator_only:
            self.discriminator.train()

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        if operating_point is not None:
            source_rate = self._batch_sample_rate(source_sample_rate)
            if source_rate is None:
                raise ValueError("SFI batches must provide the source sampling rate")
            sample_rate = operating_point.sample_rate
            latency_id = operating_point.latency_id
            noisy, clean = self._resample_pair(
                noisy, clean, source_rate, sample_rate
            )
            stft_args = get_stft_args_from_config(
                self.args.model, sample_rate=sample_rate
            )
        else:
            sample_rate = self.reference_sample_rate
            latency_id = self._sample_latency_id()
            stft_args = self.stft_args

        noisy_com = mag_pha_stft(noisy, **stft_args)[2]
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean, **stft_args)

        clean_mag_hat, clean_pha_hat, clean_com_hat = self._forward_model(
            noisy_com, latency_id=latency_id
        )

        clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)
        clean_mag_hat_con, clean_pha_hat_con, clean_com_hat_con = mag_pha_stft(
            clean_hat, **stft_args
        )

        if not self.generator_only:
            if sample_rate != 16000:
                raise ValueError(
                    "The current MetricGAN/PESQ objective is valid only at 16 kHz. "
                    "Use generator_only=true for the v2 multi-rate pilot."
                )
            one_labels = torch.ones(noisy.shape[0]).to(self.device)

            # Discriminator training
            clean_list = list(clean.cpu().numpy())
            clean_list_hat = list(clean_hat.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(
                clean_list, clean_list_hat, workers=self.num_workers
            )

            self.optim_disc.zero_grad()

            metric_r = self.discriminator(
                clean_mag.unsqueeze(1), clean_mag.unsqueeze(1)
            )
            metric_g = self.discriminator(
                clean_mag.unsqueeze(1), clean_mag_hat_con.detach().unsqueeze(1)
            )

            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(
                    batch_pesq_score.to(self.device), metric_g.flatten()
                )
            else:
                loss_disc_g = 0

            loss_disc = loss_disc_r + loss_disc_g
            loss_disc.backward()

            max_grad_norm = getattr(self.args, 'max_grad_norm', 5.0)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=max_grad_norm
            )

            self.optim_disc.step()
        else:
            loss_disc = clean_mag_hat.new_zeros(())
            max_grad_norm = getattr(self.args, 'max_grad_norm', 5.0)

        # Generator training
        self.optim.zero_grad()

        loss_magnitude = F.mse_loss(clean_mag, clean_mag_hat)
        loss_phase = phase_losses(clean_pha, clean_pha_hat)
        loss_complex = F.mse_loss(clean_com, clean_com_hat) * 2
        loss_consistency = F.mse_loss(clean_com_hat, clean_com_hat_con) * 2

        if self.generator_only:
            loss_metric = clean_mag_hat.new_zeros(())
        else:
            metric_g = self.discriminator(
                clean_mag.unsqueeze(1), clean_mag_hat_con.unsqueeze(1)
            )
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

        loss_gen = loss_metric * self.loss.metric + \
                   loss_complex * self.loss.complex + \
                   loss_consistency * self.loss.consistency + \
                   loss_magnitude * self.loss.magnitude + \
                   loss_phase * self.loss.phase

        loss_gen.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

        self.optim.step()

        loss_dict = {
            "Metric": loss_metric.item(),
            "Complex": loss_complex.item(),
            "Consistency": loss_consistency.item(),
            "Phase": loss_phase.item(),
            "Magnitude": loss_magnitude.item(),
            "Disc": loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc,
            "Gen": loss_gen.item()
        }

        return loss_dict

    def _sample_latency_id(self):
        if not self.is_latency_expert:
            return None
        return random.choice(self.latency_ids)

    def _forward_model(self, noisy_com, latency_id=None):
        if self.is_latency_expert:
            return self.model(noisy_com, latency_id=latency_id)
        return self.model(noisy_com)

    def _run_validation(self, steps):
        self.model.eval()
        if not self.generator_only:
            self.discriminator.eval()

        if self.validation_sample_rate != 16000:
            raise ValueError(
                "The current checkpoint-selection PESQ path is 16 kHz only; "
                "set validation_sample_rate=16000 until the URGENT evaluator is connected."
            )
        validation_stft_args = get_stft_args_from_config(
            self.args.model,
            sample_rate=self.validation_sample_rate if self.sfi_enabled else None,
        )

        latency_ids = self.latency_ids if self.is_latency_expert else [None]
        validation_results = {}

        for latency_id in latency_ids:
            tag = latency_id or "default"
            val_err_complex = 0
            val_err_mag = 0
            val_err_phase = 0
            clean_list, clean_hat_list = [], []

            with torch.no_grad():
                for data in self.va_loader:
                    noisy, clean = data[0], data[1]
                    source_sample_rate = data[2] if self.sfi_enabled else None
                    noisy = noisy.squeeze(0).to(self.device)
                    clean = clean.squeeze(0).to(self.device)

                    # Full-length inference (no segmentation)
                    noisy_in = noisy.unsqueeze(0)   # [1, length]
                    clean_in = clean.unsqueeze(0)    # [1, length]

                    if self.sfi_enabled:
                        source_rate = self._batch_sample_rate(source_sample_rate)
                        noisy_in, clean_in = self._resample_pair(
                            noisy_in,
                            clean_in,
                            source_rate,
                            self.validation_sample_rate,
                        )
                        clean = clean_in.squeeze(0)

                    noisy_com = mag_pha_stft(noisy_in, **validation_stft_args)[2]
                    clean_mag, clean_pha, clean_com = mag_pha_stft(
                        clean_in, **validation_stft_args
                    )

                    clean_mag_hat, clean_pha_hat, clean_com_hat = self._forward_model(
                        noisy_com, latency_id=latency_id
                    )

                    clean_hat = mag_pha_istft(
                        clean_mag_hat, clean_pha_hat, **validation_stft_args
                    )

                    # Align lengths (defensive)
                    min_len = min(clean.shape[-1], clean_hat.shape[-1])
                    clean = clean[..., :min_len]
                    clean_hat = clean_hat[..., :min_len]

                    clean_list.append(clean.squeeze().detach().cpu().numpy())
                    clean_hat_list.append(clean_hat.squeeze().detach().cpu().numpy())

                    val_err_complex += F.l1_loss(clean_com, clean_com_hat).item()
                    val_err_mag += F.l1_loss(clean_mag, clean_mag_hat).item()
                    val_err_phase += torch.mean(anti_wrapping_function(clean_pha - clean_pha_hat)).item()

            val_err_complex /= len(self.va_loader)
            val_err_mag /= len(self.va_loader)
            val_err_phase /= len(self.va_loader)
            val_pesq_result = batch_pesq(
                clean_list, clean_hat_list, workers=self.num_workers, normalize=False
            )
            if val_pesq_result is not None:
                val_pesq_score = val_pesq_result.mean().item()
            else:
                val_pesq_score = 0

            validation_results[tag] = {
                "complex": val_err_complex,
                "magnitude": val_err_mag,
                "phase": val_err_phase,
                "pesq": val_pesq_score,
            }

        pesq_scores = [v["pesq"] for v in validation_results.values()]
        val_pesq_mean = sum(pesq_scores) / len(pesq_scores)
        val_pesq_min = min(pesq_scores)

        self.logger.info("-" * 70)
        for tag, result in validation_results.items():
            self.logger.info(
                f"Validation | Step {steps} | {tag} | "
                f"Complex Diff {result['complex']:.5f} | "
                f"Magnitude Diff {result['magnitude']:.5f} | "
                f"Phase Diff {result['phase']:.5f} | "
                f"Valid PESQ {result['pesq']:.5f}"
            )
            prefix = f"Validation/{tag}" if self.is_latency_expert else "Validation"
            self.writer.add_scalar(f"{prefix}/Complex_Loss", result["complex"], steps)
            self.writer.add_scalar(f"{prefix}/Magnitude_Loss", result["magnitude"], steps)
            self.writer.add_scalar(f"{prefix}/Phase_Loss", result["phase"], steps)
            self.writer.add_scalar(f"{prefix}/Validation_PESQ_Score", result["pesq"], steps)

        if self.is_latency_expert:
            self.logger.info(
                f"Validation | Step {steps} | valid_pesq_mean {val_pesq_mean:.5f} | "
                f"valid_pesq_min {val_pesq_min:.5f}"
            )
            self.writer.add_scalar("Validation/valid_pesq_mean", val_pesq_mean, steps)
            self.writer.add_scalar("Validation/valid_pesq_min", val_pesq_min, steps)
        self.logger.info("-" * 70)

        return val_pesq_mean
