from dataclasses import dataclass


@dataclass
class Config:
    n_embedding: int = 768
    batch_size: int = 4
    vocab_size: int = 50304
    n_head: int = 6
    block_size: int = 1024
    d_k: int = 128
    dropout: float = 0.1
    bias: bool = True
    n_layers: int = 12
    aspect_ratio: int = 64
    use_flash_attention: bool = False

    # inference parameters
    use_cache: bool = True

    # training parameters
    grad_clip: float = 1.0
    learning_rate_warmup_epochs: int = 500
    learning_rate_decay_epochs: int = 4000
    evaluation_epochs: int = 1000
    training_epochs: int = 5000
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: int = 0.1
    accumulation_steps: int = 396

    # sft parameters
    sft_epochs: int = 1000

    # supervised fine-tuning
    step_to_start_from: int = 100
    model_tag: str = "model.pt"

    # RLHF
    sample_group_size: int = 10
    grpo_steps: int = 50
    clip_epsilon: float = 0.2
    kl_divergence_beta: float = 0.5


def scale(depth):
    config = Config()
    base_dimension = depth * config.aspect_ratio
    config.n_layers = depth
    # round it so its a multiple of d_k
    config.n_embedding = (base_dimension + config.d_k - 1) // config.d_k
    config.n_head = config.n_embedding // config.n_head
    return config
