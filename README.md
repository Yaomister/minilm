# MiniLM

A small GPT-2 style language model built from scratch in PyTorch, covering the full pipeline: data collection, tokenization, pretraining, supervised fine-tuning, and RL with GRPO.

## Features

- **Model**: decoder-only transformer with tied embeddings, KV cache for generation, optional flash attention, and depth-based autoscaling (`config.scale`)
- **Tokenizer**: byte-level BPE trained from scratch, using a GPT-4 style split regex
- **Data**: Wikipedia scraper that builds the raw text corpus
- **Pretraining**: DDP + `torch.compile`, Muon for weight matrices and AdamW for everything else, warmup + cosine LR schedule, gradient accumulation, MFU tracking, optional FP8
- **SFT**: fine-tunes from a pretraining checkpoint
- **GRPO**: group-relative policy optimization on GSM8K, with clipping and a KL penalty
- **Checkpointing**: save and resume model, optimizer, and metadata

## Project structure

```
config.py            model and training hyperparameters
model/               transformer and reward model
data/                scraper, BPE tokenizer, dataloader, datasets
training/            pretraining.py, sft.py, grpo.py
utils/               checkpointing, DDP setup, helpers
scripts/run.sh       SLURM job script (single A100)
```

## Usage

```bash
pip install -r requirements.txt

# pretrain (single GPU or multi-GPU with torchrun)
torchrun --nproc_per_node=<gpus> -m training.pretraining

# fine-tune from a checkpoint
python -m training.sft --checkpoint-dir <dir> --step <step>

# GRPO
python -m training.grpo --checkpoint-dir <dir> --step <step>
```

Hyperparameters are in [config.py](config.py).


