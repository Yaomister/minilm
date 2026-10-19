import os
import re
import json
import torch
import logging
from model import Model
from config import Config

logger = logging.getLogger(__name__)


def save_checkpoint(dir, step, model_data, optimizer_data, meta_data, ddp_rank):
    """Checkpointing training progress."""
    if ddp_rank == 0:
        os.makedirs(dir, exist_ok=True)

        model_path = os.path.join(dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to {model_path}.")

        meta_data_path = os.path.join(dir, f"meta_data_{step:06d}.json")
        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved meta data to {meta_data_path}.")

    if optimizer_data is not None:
        os.makedirs(dir, exist_ok=True)
        optimizer_path = os.path.join(dir, f"optimizer_{step:06d}_{ddp_rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer parameters to {optimizer_path}.")


def find_last_step(checkpoint_dir):
    """Find the last training step that was saved."""
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if re.search(r"model_(\d+)\.pt$", f)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    return max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)


def build_model(checkpoint_dir, step, device, load_optimizer=False):
    """Construct the model from the checkpointed metadata."""
    if step is None:
        step = find_last_step(checkpoint_dir)

    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, step, load_optimizer, device
    )

    # rebuild the exact architecture the checkpoint was trained with
    config = Config(**meta_data["config"])
    model = Model(config).to(device)
    model.load_state_dict(model_data)

    return model, optimizer_data, meta_data


def load_checkpoint(dir, step, load_optimizer, device, ddp_rank=0):
    """Load the checkpointed training progress."""
    model_path = os.path.join(dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(dir, f"optimizer_{step:06d}_{ddp_rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)

    meta_data_path = os.path.join(dir, f"meta_data_{step:06d}.json")
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data
