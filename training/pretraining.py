import os
import math
import time
import torch
import logging
import argparse
import contextlib
import numpy as np
from model import Model
from config import scale
from utils.setup import ddp
from dataclasses import asdict
from torch.optim import AdamW, Muon
from data.tokenizer import Tokenizer
from data.dataloader import DataLoader
from utils.common import print0, tokenize_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.common import get_peak_flops, get_device_type, synchronize
from utils.checkpoint import save_checkpoint, load_checkpoint, find_last_step
from utils.directories import (
    PRETRAINING_BIN_DATASET_DIR,
    PRETRAINING_BIN_VALIDATION_DATASET_DIR,
    PRETRAINING_RAW_DATASET_DIR,
    PRETRAINING_MERGES_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", default="checkpoints", required=False)
parser.add_argument(
    "--resume_from_checkpoint", action="store_true", required=False
)
# can only use fp8 precision on H100 GPUs or later
parser.add_argument("--use-fp8", type=bool, default=False, required=False)
# depth for autoscaling, just the number of layers in the model
parser.add_argument("--depth", type=int, default=1, required=False)

args = parser.parse_args()

can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0


config = scale(args.depth)

model = Model(config).to(device)
# keep a handle on the plain model so its state dict has no ddp / compile prefixes
raw_model = model
if can_use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
model = torch.compile(model=model, dynamic=False)

device_type = get_device_type()
flops_per_token = raw_model.estimate_flops()
peak_flops = get_peak_flops(
    torch.cuda.get_device_name() if torch.cuda.is_available() else device_type
)


@torch.no_grad()
def evaluate_loss(model, evaluation_loader):
    out = {}
    model.eval()
    for dataset, _ in {
        "validation": PRETRAINING_BIN_DATASET_DIR,
        "training": PRETRAINING_BIN_VALIDATION_DATASET_DIR,
    }.items():
        losses = torch.zeros(config.evaluation_epochs)
        for i in range(config.evaluation_epochs):
            x, y = evaluation_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[dataset] = losses.mean()

    model.train()
    return out


def configure_optimizer(model):
    # we only want to add weight decay to the weights not the biases
    to_decay = [p for p in model.parameters() if p.dim() >= 2]
    to_not_decay = [p for p in model.parameters() if p.dim() < 2]

    adam = AdamW(
        [{"params": to_not_decay, "weight_decay": 0}],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=device_type == "cuda",
    )

    muon = Muon(
        [{"params": to_decay, "weight_decay": config.weight_decay}],
        lr=config.learning_rate,
    )

    return [muon, adam]


def get_learning_rate(epoch):
    warmup_epochs, decay_epochs, learning_rate, minimum_learning_rate = (
        config.learning_rate_warmup_epochs,
        config.learning_rate_decay_epochs,
        config.learning_rate,
        config.minimum_learning_rate,
    )
    if epoch < warmup_epochs:
        return learning_rate * ((epoch + 1) / (warmup_epochs + 1))
    elif epoch > decay_epochs:
        return minimum_learning_rate
    else:
        # cosine decay
        decay_ratio = (epoch - warmup_epochs) / (decay_epochs - warmup_epochs)
        coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return minimum_learning_rate + coefficient * (
            learning_rate - minimum_learning_rate
        )


if __name__ == "__main__":
    print0("start training")

    tokenizer = Tokenizer()
    tokenizer.load(PRETRAINING_MERGES_DIR)

    if not os.path.isfile(PRETRAINING_BIN_DATASET_DIR):
        tokenize_dataset(
            tokenizer,
            PRETRAINING_RAW_DATASET_DIR,
            PRETRAINING_BIN_DATASET_DIR,
            PRETRAINING_BIN_VALIDATION_DATASET_DIR,
        )

    train_data = np.fromfile(PRETRAINING_BIN_DATASET_DIR, dtype=np.uint16)
    val_data = np.fromfile(PRETRAINING_BIN_VALIDATION_DATASET_DIR, dtype=np.uint16)

    train_loader = DataLoader(
        train_data, config.batch_size, config.block_size, ddp_rank, ddp_world_size
    )
    evaluation_loader = DataLoader(
        val_data, config.batch_size, config.block_size, ddp_rank, ddp_world_size
    )

    optimizers = configure_optimizer(model)

    if args.resume_from_checkpoint:
        last_step = find_last_step(args.checkpoint_dir)
        model_data, optimizer_data, meta_data = load_checkpoint(
            args.checkpoint_dir, last_step, True, device, ddp_rank
        )
        raw_model.load_state_dict(model_data)
        for optimizer, optimizer_state in zip(optimizers, optimizer_data):
            optimizer.load_state_dict(optimizer_state)
        starting_epoch = meta_data["step"] + 1
        total_training_time = meta_data["total_training_time"]
    else:
        starting_epoch = 0
        total_training_time = 0

    for epoch in range(starting_epoch, config.training_epochs):
        print0(f"epoch {epoch}")
        current_learning_rate = get_learning_rate(epoch)

        for optimizer in optimizers:
            for g in optimizer.param_groups:
                g["lr"] = current_learning_rate

        synchronize(device_type)
        t0 = time.time()

        # gradient accumulation so the GPUs dont explode
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for micro_step in range(config.accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            is_last = micro_step == config.accumulation_steps - 1
            # no_sync only exists on ddp models
            ctx = (
                model.no_sync()
                if can_use_ddp and not is_last
                else contextlib.nullcontext()
            )
            with ctx:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / config.accumulation_steps
                loss.backward()

        if config.grad_clip != 0:
            # gradient clipping, so one bad run doesnt throw off all the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        for optimizer in optimizers:
            optimizer.step()

        t1 = time.time()
        synchronize(device_type)

        dt = t1 - t0
        total_training_time += dt

        tokens_per_step = (
            config.batch_size
            * config.block_size
            * config.accumulation_steps
            * ddp_world_size
        )
        tokens_per_second = tokens_per_step // dt
        mfu = (
            100
            * (flops_per_token * tokens_per_step / dt)
            / (peak_flops * ddp_world_size)
        )

        print0(
            f"epoch {epoch:05d} | loss {loss:.4f} | dt {dt*1000:.1f}ms | token/s {tokens_per_second:,.0f} | mfu {mfu:.1f}%"
        )

        if (epoch + 1) % config.evaluation_intervals == 0:
            losses = evaluate_loss(model, evaluation_loader)
            print0(
                f"epoch {epoch} | train {losses['training']:.4f} | val {losses['validation']:.4f}"
            )

        # every rank has to call this on the same steps, since each rank saves its own optimizer state
        is_last_epoch = epoch == config.training_epochs - 1
        if (epoch + 1) % config.evaluation_intervals == 0 or is_last_epoch:
            save_checkpoint(
                args.checkpoint_dir,
                epoch,
                raw_model.state_dict(),
                [optimizer.state_dict() for optimizer in optimizers],
                {
                    "step": epoch,
                    "config": asdict(config),
                    "train_loss": loss.item() * config.accumulation_steps,
                    "total_training_time": total_training_time,
                },
                ddp_rank,
            )
