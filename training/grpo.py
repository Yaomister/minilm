import re
import torch
import logging
import datasets
from config import Config
from utils.setup import ddp
from dataclasses import asdict
from argparse import ArgumentParser
from data.tokenizer import Tokenizer
from utils.checkpoint import build_model
from utils.directories import SFT_MERGES_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

config = Config()
can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0

# finished sequences are padded with end of text
PAD_ID = config.vocab_size - 1


def load_dataset():
    dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    return dataset


def get_log_probabilities(policy, x, y):
    # passing targets makes the model return logits for every position
    logits, _ = policy(x, y)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    return log_probabilities.gather(-1, y.unsqueeze(-1)).squeeze(-1)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    # defaults to the latest step in the checkpoint directory
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def sample_group(policy, prompt_ids, G, max_new=256):
    prompt = prompt_ids.size(0)
    sequence = prompt_ids.unsqueeze(0).repeat(G, 1)
    done = torch.zeros(G, dtype=torch.bool, device=sequence.device)
    for _ in range(min(max_new, policy.block_size - prompt)):
        logits, _ = policy(sequence)
        next = torch.multinomial(torch.nn.functional.softmax(logits, -1), 1).squeeze(1)
        next = torch.where(done, torch.full_like(next, PAD_ID), next)
        sequence = torch.cat([sequence, next.unsqueeze(1)], dim=1)
        done |= next == PAD_ID
        if done.all():
            break

    return sequence


def extract_answer(text):
    # gsm8k answers end with "#### <number>"
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    return match.group(1).replace(",", "") if match else None


def get_rewards(response, answer, tokenizer):
    # 1 if the final answer matches, 0 otherwise
    rewards = [
        float(extract_answer(tokenizer.decode(r)) == extract_answer(answer))
        for r in response.tolist()
    ]
    return torch.tensor(rewards, device=response.device)


def calculate_advantage(responses):
    return torch.rand_like(responses)


def run(args):
    # the architecture comes from the checkpoint, the grpo hyperparameters from config.py
    reference, _, _ = build_model(args.checkpoint_dir, args.step, device)
    policy, _, _ = build_model(args.checkpoint_dir, args.step, device)

    reference.eval()
    reference.requires_grad_(False)

    # fresh optimizer, the pretraining checkpoint holds muon + adamw states that do not map onto this
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

    dataset = load_dataset()

    tokenizer = Tokenizer()
    tokenizer.load(SFT_MERGES_DIR)

    for epoch in range(config.grpo_steps):
        # each rank takes its own slice of the questions
        for row in dataset.shard(ddp_world_size, ddp_rank):
            prompt_ids = torch.tensor(
                tokenizer.encode(f"Question: {row['question']}\nAnswer:"), device=device
            )[-(policy.block_size // 2) :]
            prompt = prompt_ids.size(0)

            with torch.no_grad():
                sequence = sample_group(policy, prompt_ids, config.sample_group_size)
                responses = sequence[:, prompt:]
                rewards = get_rewards(responses, row["answer"], tokenizer)

                reward_std = torch.std(rewards, dim=-1)
                reward_mean = torch.mean(rewards, dim=-1)

                advantage = ((rewards - reward_mean) / (reward_std + 1e-8)).unsqueeze(1)

                # x predicts y one token ahead, only the response tokens count in the loss
                x, y = sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
                mask = torch.zeros_like(y, dtype=torch.bool)
                # drop the padding after the first end of text
                is_pad = (responses == PAD_ID).long()
                mask[:, prompt - 1 :] = (is_pad.cumsum(dim=1) - is_pad) == 0

                old_logp = get_log_probabilities(policy, x, y)
                ref_logp = get_log_probabilities(reference, x, y)

            new_logp = get_log_probabilities(policy, x, y)
            ratio = torch.exp(new_logp - old_logp)
            surregate = torch.min(
                ratio * advantage,
                torch.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                * advantage,
            )
            # k3 estimator of kl(policy || reference)
            log_ratio = ref_logp - new_logp
            kl_divergence = torch.exp(log_ratio) - log_ratio - 1

            loss = -(surregate - config.kl_divergence_beta * kl_divergence)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    torch.save(
        {
            "model": policy.state_dict(),
            "epoch": epoch,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "best_loss": loss.item(),
        },
        "rl_model.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
