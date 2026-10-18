import torch
from model import Model
from config import Config
from utils.setup import ddp
from dataclasses import asdict
from argparse import ArgumentParser
from datasets import load_dataset
from utils.common import get_device_type
from utils.checkpoint import load_checkpoint

config = Config()
can_use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = ddp()
master_process = ddp_rank == 0

PAD_ID = 0


def load_dataset():
    dataset = load_dataset("openai/gsm8k", split="main")
    return dataset


def get_log_probabilities(policy, x, y):
    z = policy(x)
    return torch.log(z[y])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", default=0)
    return parser.parse_args()


def sample_group(policy, prompt_ids, G, max_new=256):
    prompt = prompt_ids.size(0)
    sequence = prompt_ids.unsqueeze(0).repeat(G, 1)
    done = torch.zeros_like(G, dtype=torch.bool)
    for _ in range(min(G, max_new)):
        logits = policy(sequence)
        next = torch.multinomial(torch.nn.functional.softmax(logits, -1), 1).unsqueeze(
            1
        )
        next = torch.where(done, torch.full_like(next, PAD_ID), next)

    return done


def get_rewards(response):
    return 0


def calculate_advantage(responses):
    return torch.rand_like(responses)


def run(args):
    model_data, optimizer_data, _ = load_checkpoint(
        args.checkpoint_dir, args.step, True, device=get_device_type()
    )
    reference = Model(config)
    policy = Model(config)

    reference.load_state_dict(model_data)
    policy.load_state_dict(model_data)

    reference.eval()
    reference.requires_grad(True)

    optimizer = torch.optim.AdamW(policy.parameters, lr=1e-6)
    optimizer.load_state_dict(optimizer_data)

    dataset = load_dataset()

    for epoch in range(config.grpo_steps):
        for x, y in dataset[::ddp_rank]:
            with torch.no_grad():
                responses = sample_group(policy, x, config.sample_group_size)
                rewards = get_rewards(responses)

                reward_std = torch.std(rewards, dim=-1)
                reward_mean = torch.mean(rewards, dim=-1)

                advantage = (rewards - reward_mean) // (reward_std + 1e-8)

                old_logp = get_log_probabilities(policy, x, y)
                ref_logp = get_log_probabilities(reference, x, y)

            new_logp = get_log_probabilities(policy, x, y)
            ratio = torch.exp(new_logp - old_logp)
            surregate = min(
                ratio * advantage,
                torch.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon),
            )
            kl_divergence = torch.kl_div(ref_logp, new_logp)

            loss = -(surregate - config.kl_divergence_beta * kl_divergence)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    torch.save(
        {
            "model": policy.state_dict(),
            "epoch": epoch,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "best_loss": loss,
        },
        "rl_model.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
