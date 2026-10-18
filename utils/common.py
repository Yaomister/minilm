import os
import torch
import numpy as np


def print0(content="", **kwargs):
    if os.environ.get("RANK", 0) == 0:
        print(content)


def tokenize_dataset(tokenizer, dir, training_dataset_dir, validation_dataset_dir):
    with open(dir, "r") as f:
        raw_text = f.read()

    ids = tokenizer.encode(raw_text)

    print0(f"{len(raw_text)} characters to {len(ids)} tokens")

    split = int(len(ids) * 0.8)

    np.array(ids[:split], dtype=np.uint16).tofile(training_dataset_dir)
    np.array(ids[split:], dtype=np.uint16).tofile(validation_dataset_dir)


def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    # for cpu
    return float("inf")
