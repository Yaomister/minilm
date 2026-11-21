"""Group based relative optimization."""

import re
import torch
import logging
import datasets
import torch.distributed as dist
from config import Config
from utils.setup import ddp
from argparse import ArgumentParser
from data.tokenizer import Tokenizer
from utils.checkpoint import build_model, save_checkpoint
from utils.directories import SFT_MERGES_DIR
from utils.common import print0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

config = Config()
can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0

# finished sequences are padded with end of text
PAD_ID = config.vocab_size - 1


def load_dataset():
    """Load the GSM8K dataset."""
    dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
    return dataset


def get_log_probabilities(policy, x, y):
    """Calculate the log probability of the logits."""
    # passing targets makes the model return logits for every position
    logits, _ = policy(x, y)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    return log_probabilities.gather(-1, y.unsqueeze(-1)).squeeze(-1)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    # defaults to the latest step in the checkpoint directory
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    return parser.parse_args()


@torch.no_grad()
def sample_group(policy, prompt_ids, G, max_new=256):
    """Generate a group of responses."""

    prompt = prompt_ids.size(0)
    sequence = prompt_ids.unsqueeze(0).repeat(G, 1)
    done = torch.zeros(G, dtype=torch.bool, device=sequence.device)

    policy._reset_cache()
    logits, _ = policy(sequence, use_cache=True)

    # generate a group of responses
    for _ in range(min(max_new, policy.block_size - prompt)):
        # samples a response from the logits
        next_token = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        next_token = torch.where(done, torch.full_like(next_token, PAD_ID), next_token)

        # concat the generated next token to the original sequence
        sequence = torch.cat([sequence, next_token.unsqueeze(1)], dim=1)

        done |= next_token == PAD_ID
        if done.all():
            break

        logits, _ = policy(next_token.unsqueeze(1), use_cache=True)

    policy._reset_cache()
    return sequence


def extract_answer(text):
    """Extract the answer for grading GSM8K"""

    # gsm8k answers end with "#### <number>"
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    return match.group(1).replace(",", "") if match else None


def get_rewards(response, answer, tokenizer):
    """Get the reward for calculating the advantage."""

    # 1 if the final answer matches, 0 otherwise
    rewards = [
        float(extract_answer(tokenizer.decode(r)) == extract_answer(answer))
        for r in response.tolist()
    ]
    return torch.tensor(rewards, device=response.device)


def grpo(args):
    """Group Based Relative Optimization."""

    # the architecture comes from the checkpoint, the grpo hyperparameters from config.py
    reference, _, meta_data = build_model(args.checkpoint_dir, args.step, device)
    policy, _, _ = build_model(args.checkpoint_dir, args.step, device)

    reference.eval()
    reference.requires_grad_(False)
    policy.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

    dataset = load_dataset()
    # each rank takes its own slice of the questions
    shard = dataset.shard(ddp_world_size, ddp_rank)
    if can_use_ddp:
        shard = shard.select(range(len(dataset) // ddp_world_size))

    tokenizer = Tokenizer()
    tokenizer.load(SFT_MERGES_DIR)

    # for monitoring whether the model is getting any reward signal at all
    log_interval = 50
    questions, correct, samples, skipped, updates = 0, 0.0, 0, 0, 0
    last_loss = None

    for epoch in range(config.grpo_steps):
        for row in shard:

            if questions > 0 and questions % log_interval == 0:
                print0(
                    f"epoch {epoch} | questions {questions} | accuracy {correct / samples:.3f} "
                    f"| groups with signal {(questions - skipped) / questions:.3f} | updates {updates} "
                    f"| loss {last_loss if last_loss is None else f'{last_loss:.4f}'}"
                )

            prompt_ids = torch.tensor(
                tokenizer.encode(f"Question: {row['question']}\nAnswer:"), device=device
            )[-(policy.block_size // 2) :]
            prompt = prompt_ids.size(0)

            # generate a group of answers
            sequence = sample_group(policy, prompt_ids, config.sample_group_size)

            # get their actual answer to the question
            responses = sequence[:, prompt:]

            # get the reward
            rewards = get_rewards(responses, row["answer"], tokenizer)

            questions += 1
            correct += rewards.sum().item()
            samples += rewards.numel()

            # if every answer got the same reward the advantage is all zeros, there's nothing
            # to learn from this group so skip the forward/backward passes
            has_signal = rewards.std().item() > 0
            if not has_signal:
                skipped += 1

            ranks_with_signal = torch.tensor(float(has_signal), device=device)
            if can_use_ddp:
                dist.all_reduce(ranks_with_signal)
            ranks_with_signal = int(ranks_with_signal.item())
            if ranks_with_signal == 0:
                continue

            if has_signal:
                # calculate the z-score
                advantage = (
                    (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                ).unsqueeze(1)

                # x predicts y one token ahead, only the response tokens count in the loss
                x, y = sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
                mask = torch.zeros_like(y, dtype=torch.bool)
                # drop the padding after the first end of text
                is_pad = (responses == PAD_ID).long()
                mask[:, prompt - 1 :] = (is_pad.cumsum(dim=1) - is_pad) == 0

                with torch.no_grad():
                    old_logp = get_log_probabilities(policy, x, y)
                    ref_logp = get_log_probabilities(reference, x, y)

                new_logp = get_log_probabilities(policy, x, y)
                ratio = torch.exp(new_logp - old_logp)

                # clips the advantage so a single policy update doesnt swing too much from the reference
                surrogate = torch.min(
                    ratio * advantage,
                    torch.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                    * advantage,
                )

                log_ratio = ref_logp - new_logp

                # so the new policy doesnt land too far from the reference
                kl_divergence = torch.exp(log_ratio) - log_ratio - 1

                # the GRPO loss function
                loss = -(surrogate - config.kl_divergence_beta * kl_divergence)
                loss = (loss * mask).sum() / mask.sum()
                loss.backward()
                last_loss = loss.item()

            if can_use_ddp:
                for p in policy.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)
                    p.grad /= ranks_with_signal

            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)

            # update the policy
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            updates += 1

    save_checkpoint(
        args.output_dir,
        epoch,
        policy.state_dict(),
        None,
        {
            "step": epoch,
            "config": meta_data["config"],
            "updates": updates,
            # None if no group ever had a reward signal
            "last_loss": last_loss,
        },
        ddp_rank,
    )

    if can_use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = _parse_args()
    grpo(args)
