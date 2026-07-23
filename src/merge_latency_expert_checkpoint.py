"""
Merge independently trained LaCo-SENet checkpoints into a latency-expert model.

The merged checkpoint is intended for latency-expert experiments:
- shared modules are initialized from the selected shared checkpoint
  (L3 by default for encoder_decoder, L0 by default for decoder_only)
- latency-specific branches are initialized from their corresponding
  single-latency checkpoints
- supported scopes are dsddb_bn, dsddb_only, asymconv_only, and full,
  controlled by the config's model.latency_expert.expert_scope field
- supported topologies are encoder_decoder and decoder_only, controlled by
  the config's model.latency_expert.topology field

Example:
    python -m src.merge_latency_expert_checkpoint \\
      --config conf/experiments/latency_expert_dsddb_bn.yaml \\
      --output_dir results/experiments/latency_expert_dsddb_bn/s2039 \\
      --L0 checkpoints/M1_12.5ms/s2039/model_279000.th \\
      --L1 checkpoints/M2_25.0ms/s2039/model_147000.th \\
      --L3 checkpoints/M4_50.0ms/s2039/model_97000.th \\
      --L5 checkpoints/M6_75.0ms/s2039/model_163000.th
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Mapping

import torch
from omegaconf import OmegaConf

from src.utils import load_model


def load_experiment_config(config_path: str):
    """Load either a complete YAML config or a Hydra experiment config."""
    conf = OmegaConf.load(config_path)
    if conf.get("model") is not None and conf.model.get("win_len") is not None:
        return conf

    path = Path(config_path).resolve()
    conf_dir = None
    for parent in [path.parent, *path.parents]:
        if parent.name == "conf":
            conf_dir = parent
            break
    if conf_dir is None:
        raise ValueError(
            f"Config {config_path} is not a complete config and is not under a conf/ directory."
        )

    config_name = path.relative_to(conf_dir).with_suffix("").as_posix()
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        return compose(config_name=config_name)


def resolve_checkpoint_file(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(path)

    states_path = p / "states.th"
    if not states_path.exists():
        raise FileNotFoundError(f"Directory checkpoint requires states.th: {p}")

    states = torch.load(states_path, map_location="cpu", weights_only=False)
    best = states["best_models"][0]
    ckpt_path = p / f"model_{best['steps']}.th"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
    return ckpt_path


def load_model_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt_path = resolve_checkpoint_file(path)
    package = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in package:
        raise KeyError(f"Checkpoint does not contain a 'model' state: {ckpt_path}")
    return package["model"]


def _copy_key(dst, dst_key: str, src, src_key: str, used_sources: set, source_label: str):
    if src_key not in src:
        raise KeyError(f"Missing {source_label} source key for {dst_key}: {src_key}")
    if dst[dst_key].shape != src[src_key].shape:
        raise ValueError(
            f"Shape mismatch for {dst_key}: target {tuple(dst[dst_key].shape)} "
            f"vs {source_label} {src_key} {tuple(src[src_key].shape)}"
        )
    dst[dst_key] = src[src_key].clone()
    used_sources.add(src_key)


def merge_states(
    expert_state: Mapping[str, torch.Tensor],
    shared_state: Mapping[str, torch.Tensor],
    branch_states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    merged = {k: v.clone() for k, v in expert_state.items()}
    assigned = set()
    used_sources = {latency_id: set() for latency_id in branch_states}
    used_shared = set()

    for key in list(merged.keys()):
        # Latency-specific branches:
        # dense_encoder.dense_block.experts.L3.* -> dense_encoder.dense_block.*
        # dense_encoder.dense_block.dense_block.0.0.experts.L3.* ->
        #     dense_encoder.dense_block.dense_block.0.0.*
        matched_branch = False
        for latency_id, source in branch_states.items():
            marker = f".experts.{latency_id}."
            if marker in key:
                src_key = key.replace(marker, ".")
                _copy_key(merged, key, source, src_key, used_sources[latency_id], latency_id)
                assigned.add(key)
                matched_branch = True
                break
        if matched_branch:
            continue

        # Latency-specific BN outside DSDDB (dsddb_bn only):
        # dense_encoder.dense_conv_1.1.norms.L3.weight -> dense_encoder.dense_conv_1.1.weight
        matched_bn = False
        for latency_id, source in branch_states.items():
            marker = f".norms.{latency_id}."
            if marker in key:
                src_key = key.replace(marker, ".")
                _copy_key(merged, key, source, src_key, used_sources[latency_id], latency_id)
                assigned.add(key)
                matched_bn = True
                break
        if matched_bn:
            continue

        # Shared modules with identical state_dict keys.
        if key in shared_state:
            _copy_key(merged, key, shared_state, key, used_shared, "shared")
            assigned.add(key)

    unassigned = sorted(set(merged) - assigned)
    if unassigned:
        raise RuntimeError(
            "Some expert checkpoint keys were not initialized from source checkpoints:\n"
            + "\n".join(unassigned[:50])
            + ("\n..." if len(unassigned) > 50 else "")
        )

    return merged


def save_merged_checkpoint(output_dir: Path, conf, state: Dict[str, torch.Tensor]):
    output_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(conf, hydra_dir / "config.yaml")

    model_package = {"model": state}
    torch.save(model_package, output_dir / "model_0.th")

    states_package = {
        "model": state,
        "best_models": [
            {
                "steps": 0,
                "valid_pesq_value": 0.0,
                "model": state,
            }
        ],
        "step": 0,
        "args": conf,
    }
    torch.save(states_package, output_dir / "states.th")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Latency expert config YAML")
    parser.add_argument("--output_dir", required=True, help="Output checkpoint directory")
    parser.add_argument("--shared_latency_id", default=None,
                        help=(
                            "Latency id whose checkpoint initializes shared modules. "
                            "Defaults to L3 for encoder_decoder and L0 for decoder_only."
                        ))
    parser.add_argument("--L0", required=True, help="L0 checkpoint file or run directory")
    parser.add_argument("--L1", required=True, help="L1 checkpoint file or run directory")
    parser.add_argument("--L3", required=True, help="L3 checkpoint file or run directory")
    parser.add_argument("--L5", required=True, help="L5 checkpoint file or run directory")
    return parser


def main():
    args = build_parser().parse_args()
    conf = load_experiment_config(args.config)
    conf.model.type = "latency_expert_backbone"

    branch_paths = {
        "L0": args.L0,
        "L1": args.L1,
        "L3": args.L3,
        "L5": args.L5,
    }
    latency_expert_cfg = conf.model.get("latency_expert", {})
    topology = latency_expert_cfg.get("topology", latency_expert_cfg.get("expert_topology", "encoder_decoder"))
    if args.shared_latency_id is None:
        args.shared_latency_id = "L0" if topology == "decoder_only" else "L3"
    if args.shared_latency_id not in branch_paths:
        raise ValueError(f"shared_latency_id must be one of {sorted(branch_paths)}")

    model = load_model(conf.model, device="cpu")
    expert_state = model.state_dict()
    branch_states = {
        latency_id: load_model_state(path)
        for latency_id, path in branch_paths.items()
    }
    shared_state = branch_states[args.shared_latency_id]

    merged = merge_states(expert_state, shared_state, branch_states)
    model.load_state_dict(merged, strict=True)

    save_merged_checkpoint(Path(args.output_dir), conf, merged)
    total_params = sum(v.numel() for v in merged.values())
    print(f"Merged latency expert checkpoint saved to: {os.path.abspath(args.output_dir)}")
    print(f"State tensors: {len(merged)} | elements: {total_params}")


if __name__ == "__main__":
    main()
