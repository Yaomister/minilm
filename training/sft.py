import os
import torch
import logging
import datasets
import numpy as np
from config import Config
from utils.setup import ddp
from utils.directories import (
    SFT_BIN_DATASET_DIR,
    SFT_BIN_VALIDATION_DATASET_DIR,
    SFT_MERGES_DIR,
    SFT_RAW_DATASET_DIR,
)
from argparse import ArgumentParser
from data.tokenizer import Tokenizer
from data.dataloader import DataLoader
from utils.checkpoint import build_model, save_checkpoint
from utils.common import tokenize_dataset, print0

# set up the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

config = Config()
can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    # defaults to the latest step in the checkpoint directory
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints/sft")
    return parser.parse_args()


def write_raw_dataset():
    """Write the GSM8K questions in the question/answer format."""
    os.makedirs(os.path.dirname(SFT_RAW_DATASET_DIR), exist_ok=True)
    dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    with open(SFT_RAW_DATASET_DIR, "w") as f:
        for row in dataset:
            f.write(
                f"Question: {row['question']}\nAnswer: {row['answer']}<|endoftext|>"
            )


@torch.no_grad()
def evaluate_loss(model, validation_loader):
    model.eval()
    losses = torch.zeros(config.evaluation_epochs)
    for i in range(config.evaluation_epochs):
        x, y = validation_loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean()


def sft(args):
    """Supervised fine-tuning."""

    # grab checkpoint
    model, _, meta_data = build_model(args.checkpoint_dir, args.step, device)

    # set up the tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(SFT_MERGES_DIR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_learning_rate)

    if not os.path.isfile(SFT_RAW_DATASET_DIR):
        write_raw_dataset()

    if not os.path.isfile(SFT_BIN_DATASET_DIR):
        tokenize_dataset(
            tokenizer,
            SFT_RAW_DATASET_DIR,
            SFT_BIN_DATASET_DIR,
            SFT_BIN_VALIDATION_DATASET_DIR,
        )

    train_dataset = np.fromfile(SFT_BIN_DATASET_DIR, dtype=np.uint16)
    validation_dataset = np.fromfile(SFT_BIN_VALIDATION_DATASET_DIR, dtype=np.uint16)

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

    model.train()
    for epoch in range(config.sft_epochs):
        optimizer.zero_grad()
        for _ in range(config.sft_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            loss = loss / config.sft_accumulation_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        print0(
            f"epoch {epoch:05d} | loss {loss.item() * config.sft_accumulation_steps:.4f}"
        )

        if (epoch + 1) % config.evaluation_intervals == 0:
            print0(f"epoch {epoch} | val {evaluate_loss(model, validation_loader):.4f}")

    save_checkpoint(
        args.output_dir,
        epoch,
        model.state_dict(),
        None,
        {"step": epoch, "config": meta_data["config"]},
        ddp_rank,
    )


if __name__ == "__main__":
    args = _parse_args()
    sft(args)
