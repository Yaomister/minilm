import os
import torch
import logging
import numpy as np
from pathlib import Path
from config import Config
from utils.setup import ddp
from torch.optim import AdamW
from utils.directories import (
    SFT_BIN_DATASET_DIR,
    SFT_BIN_VALIDATION_DATASET_DIR,
    SFT_MERGES_DIR,
    SFT_RAW_DATASET_DIR,
)
from argparse import ArgumentParser
from data.tokenizer import Tokenizer
from data.dataloader import DataLoader
from utils.checkpoint import build_model
from utils.common import tokenize_dataset, print0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

config = Config()
can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    # defaults to the latest step in the checkpoint directory
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def sft(args):

    # the architecture comes from the checkpoint, the sft hyperparameters from config.py
    model, _, meta_data = build_model(args.checkpoint_dir, args.step, device)

    tokenizer = Tokenizer()

    # fresh optimizer, the pretraining checkpoint holds muon + adamw states that do not map onto this
    optimizer = torch.optim.AdamW(model.parameters)

    if not os.path.isfile(SFT_BIN_DATASET_DIR):
        tokenize_dataset(
            tokenizer,
            SFT_RAW_DATASET_DIR,
            SFT_BIN_DATASET_DIR,
            SFT_BIN_VALIDATION_DATASET_DIR,
        )

    train_dataset = np.fromfile(SFT_BIN_DATASET_DIR, dtype=np.float16)
    validation_dataset = np.fromfile(SFT_BIN_VALIDATION_DATASET_DIR, dtype=np.float16)

    train_loader = DataLoader(
        train_dataset,
        config.batch_size,
        model.block_size,
        ddp_rank,
        ddp_world_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        config.batch_size,
        model.block_size,
        ddp_rank,
        ddp_world_size,
    )

    for epoch in range(config.sft_epochs):
        for minibatch in range(config.accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to_device(device), y.to_device(device)

            logits, loss = model(x, y)
            loss = loss / config.accumulation_steps
            loss.backwards()

        optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    sft(args)
