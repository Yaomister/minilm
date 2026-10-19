import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embedding = config.n_embedding
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_layers = config.n_layers
        self.use_cache = config.use_cache
        self.eot_token_id = config.vocab_size - 1
        self.current_position = 0

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embedding),
                wpe=nn.Embedding(config.block_size, config.n_embedding),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=LayerNormalization(config),
            )
        )

        self.lm_head = nn.Linear(config.n_embedding, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _reset_cache(self):
        for block in self.transformer.h:
            block.attention._reset_cache()
        self.current_position = 0

    @torch.no_grad()
    def generate(self, indexes, maximum_new_tokens):
        self._reset_cache()

        if self.use_cache:
            logits, _ = self.forward(indexes[:, -self.block_size :], use_cache=True)
            for _ in range(maximum_new_tokens):
                next_index = logits.argmax(dim=-1, keepdim=True)
                indexes = torch.cat([indexes, next_index], dim=1)
                if (next_index == self.eot_token_id).all():
                    break
                logits, _ = self.forward(next_index, use_cache=True)
        else:
            for _ in range(maximum_new_tokens):
                logits, _ = self.forward(indexes)
                next_index = logits.argmax(dim=-1, keepdim=True)
                indexes = torch.concat([indexes, next_index], dim=1)

        return indexes

    def estimate_flops(self):
        # 2 for forward pass and 4 for backward pass per parameter, and 12 * number of layers * embedding size * block size
        parameters = sum(p.numel() for p in self.parameters())
        return 6 * parameters + 12 * self.n_embedding * self.block_size * self.n_layers

    def forward(self, x, targets=None, use_cache=False):

        _, sequence_length = x.size()

        # the kv cache is only for generation, training and evaluation always run without it
        if use_cache:
            p = torch.arange(
                self.current_position,
                self.current_position + sequence_length,
                dtype=torch.long,
                device=x.device,
            )
            self.current_position += sequence_length
        else:
            p = torch.arange(0, sequence_length, dtype=torch.long, device=x.device)

        token_embeddings = self.transformer.wte(x)
        position_embeddings = self.transformer.wpe(p)

        x = self.transformer.drop(token_embeddings + position_embeddings)

        for block in self.transformer.h:
            x = block(x, use_cache)

        x = self.transformer.ln_f(x)

        # training
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)


class LayerNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embedding))
        self.bias = nn.Parameter(torch.zeros(config.n_embedding))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embedding % config.n_head == 0

        self.c_attention = nn.Linear(
            config.n_embedding, config.n_embedding * 3, bias=config.bias
        )

        self.c_proj = nn.Linear(
            config.n_embedding, config.n_embedding, bias=config.bias
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_embedding = config.n_embedding
        self.d_k = config.d_k
        self.dropout = config.dropout
        self.n_heads = config.n_head
        self.use_flash_attention = config.use_flash_attention

        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)

    def _reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        batch_size, sequence_length, _ = x.size()

        q, k, v = self.c_attention(x).split(self.n_embedding, dim=-1)

        if use_cache:
            if self.cache_k is None and self.cache_v is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.concat([self.cache_k, k], dim=1)
                self.cache_v = torch.concat([self.cache_v, v], dim=1)

            k = self.cache_k
            v = self.cache_v

        # (sequence_length, sequence_length)
        mask = torch.triu(
            torch.ones(
                sequence_length,
                k.size(1),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=(k.size(1) - sequence_length + 1),
        )

        # sawpping dimension 1 and 2 so the score calculated is per head, and you end with a tensor that is (sequence_length, d_k)
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, v.size(1), self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k.size(1), self.n_heads, self.d_k).transpose(1, 2)

        if self.use_flash_attention:
            # PyTorch uses FlashAttention2
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=~mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # (sequence_length, sequence_length)
            attention = q @ k.transpose(-2, -1)

            attention = attention.masked_fill(mask, float("-inf"))
            attention = attention / math.sqrt(self.d_k)
            attention = F.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)
            # (batch_size, n_head, sequence_length, d_k)
            y = attention @ v

        y = y.transpose(1, 2).reshape(batch_size, sequence_length, self.n_embedding)

        y = self.residual_dropout(self.c_proj(y))

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = LayerNormalization(config)
        self.ln_2 = LayerNormalization(config)

    def forward(self, x, use_cache=False):
        x = x + self.attention(self.ln_1(x), use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embedding, config.n_embedding * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.n_embedding * 4, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
