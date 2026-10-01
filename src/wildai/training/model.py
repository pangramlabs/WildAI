# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""The GPT every WildAI model uses (nanochat's modded recipe, the only variant the paper trains).

Rotary embeddings, QK-norm (with 1.2x sharpening), untied input embedding and output head, ReLU^2 MLP, RMSNorm without
learnable parameters (also right after the token embedding), no biases, a sliding-window pattern (`SSSL`: three
quarter-context layers, then one full-context layer; the last layer is always full), ResFormer-style gated value
embeddings on alternating layers, a "smear" of the previous token's embedding, a mid-network "backout" subtraction,
per-layer residual and input-embedding scalars, and a tanh logit soft-cap at 15.

Importing this module has no side effects. Build a model with `GPT(config)` on the meta device, move it with
`to_empty(device=...)`, then call `init_weights(...)` (which also loads Flash Attention 3 on Hopper GPUs). To load a
trained checkpoint use `wildai.training.checkpoint.load_model`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wildai.training import attention

VALUE_EMBED_GATE_CHANNELS = 12
SMEAR_GATE_CHANNELS = 24
LOGIT_SOFTCAP = 15.0
ROTARY_BASE = 100_000

# Released checkpoints record the recipe's switches in their model config; the model only exists with these values.
RECIPE_SWITCHES: dict[str, Any] = {
    "mlp_act": "relu2",
    "qk_norm": True,
    "value_embeds": True,
    "smear": True,
    "backout": True,
    "layer_lambdas": True,
    "logit_softcap": True,
    "embed_norm": True,
    "init_style": "nanochat",
}


@dataclass(frozen=True)
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GPTConfig:
        """Read a checkpoint's `model_config`, rejecting checkpoints trained with a different recipe."""
        for key, expected in RECIPE_SWITCHES.items():
            if key in values and values[key] != expected:
                raise ValueError(f"checkpoint uses {key}={values[key]!r}; this code only implements {key}={expected!r}")
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass(frozen=True)
class ParamCounts:
    wte: int
    value_embeds: int
    lm_head: int
    transformer_matrices: int
    scalars: int

    @property
    def total(self) -> int:
        return self.wte + self.value_embeds + self.lm_head + self.transformer_matrices + self.scalars

    @property
    def paper_n(self) -> int:
        """The paper's N: every parameter except the value-embedding tables (the input embedding is counted)."""
        return self.total - self.value_embeds

    @property
    def scaling_params(self) -> int:
        """nanochat's scaling-parameter count (transformer matrices + output head), used to scale weight decay."""
        return self.transformer_matrices + self.lm_head


def compute_dtype_for(device: torch.device) -> torch.dtype:
    """bf16 on Ampere or newer GPUs, fp32 elsewhere."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0):
        return torch.bfloat16
    return torch.float32


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Keeps fp32 master weights but runs the matmul in the activation dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """Alternating layers, always including the last."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class KVCache:
    """Per-layer key/value cache in `(B, T, H, D)` layout, plus the previous token's embedding for the smear."""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (config.n_layer, batch_size, max_len, config.n_kv_head, config.head_dim)
        self.n_layers = config.n_layer
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding: torch.Tensor | None = None

    def position(self) -> int:
        return int(self.cache_seqlens[0].item())

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        self.cache_seqlens += num_tokens


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        assert config.n_embd % config.n_head == 0 and config.n_head % config.n_kv_head == 0
        self.c_q = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        gated = has_value_embed(layer_idx, config.n_layer)
        self.ve_gate = Linear(VALUE_EMBED_GATE_CHANNELS, self.n_kv_head, bias=False) if gated else None

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        B, T, _C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :VALUE_EMBED_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        if kv_cache is None:
            y = attention.causal_attention(q, k, v, window_size)
        else:
            k_cache, v_cache = kv_cache.layer(self.layer_idx)
            y = attention.cached_attention(q, k_cache, v_cache, k, v, kv_cache.cache_seqlens, window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    """Construct on the meta device; real initialisation happens in `init_weights`."""

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = torch.float32
        self.window_sizes = self._window_sizes(config)
        padded_vocab = math.ceil(config.vocab_size / pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(SMEAR_GATE_CHANNELS, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        kv_dim = config.n_kv_head * config.head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab, kv_dim) for i in range(config.n_layer) if has_value_embed(i, config.n_layer)})
        # Rotary tables cover 10x the training context so generation can run past it; not saved in checkpoints.
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._rotary(self.rotary_seq_len, config.head_dim, torch.device("meta"), torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _window_sizes(config: GPTConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        assert pattern and set(pattern) <= {"S", "L"}, f"invalid window pattern {pattern!r}"
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # quarter context, rounded up to a multiple of 128 (2048 -> 512)
        sizes = [(long_window if pattern[i % len(pattern)] == "L" else short_window, 0) for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)
        return sizes

    @staticmethod
    def _rotary(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (ROTARY_BASE ** (channels / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
        return freqs.cos().to(dtype)[None, :, None, :], freqs.sin().to(dtype)[None, :, None, :]

    @torch.no_grad()
    def init_weights(self, compute_dtype: torch.dtype | None = None) -> None:
        """Initialise every tensor on the model's current device (seed the RNG first for reproducible weights)."""
        device = self.transformer.wte.weight.device
        self.compute_dtype = compute_dtype or compute_dtype_for(device)
        attention.prepare(device)
        n_embd, n_layer = self.config.n_embd, self.config.n_layer
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * n_embd**-0.5  # uniform bound with the standard deviation of N(0, 1/n_embd)
        for block in self.transformer.h:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.c_proj.weight)
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        nn.init.zeros_(self.smear_lambda)
        nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        nn.init.constant_(self.backout_lambda, 0.2)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        self.cos, self.sin = self._rotary(self.rotary_seq_len, self.config.head_dim, device, self.compute_dtype)
        # Embedding tables are stored in the compute dtype; the optimizer tolerates it and it halves their memory.
        self.transformer.wte.to(dtype=self.compute_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=self.compute_dtype)

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def param_counts(self) -> ParamCounts:
        scalars = sum(p.numel() for p in (self.resid_lambdas, self.x0_lambdas, self.smear_gate.weight, self.smear_lambda, self.backout_lambda))
        return ParamCounts(
            wte=self.transformer.wte.weight.numel(),
            value_embeds=sum(p.numel() for p in self.value_embeds.parameters()),
            lm_head=self.lm_head.weight.numel(),
            transformer_matrices=sum(p.numel() for p in self.transformer.h.parameters()),
            scalars=scalars,
        )

    def estimate_flops(self) -> int:
        """Training FLOPs per token: 6 per matmul weight plus attention scores (window-limited per layer)."""
        counts = self.param_counts()
        matmul_params = counts.total - counts.wte - counts.value_embeds - counts.scalars
        h, q, t = self.config.n_head, self.config.head_dim, self.config.sequence_len
        attn = sum(12 * h * q * min(window, t) for window, _ in self.window_sizes)
        return 6 * matmul_params + attn

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        """Logits `(B, T, vocab)` without targets; cross-entropy (ignoring target -1) with targets."""
        _B, T = idx.size()
        assert self.cos.dtype == self.compute_dtype, "call init_weights() before running the model"
        t0 = 0 if kv_cache is None else kv_cache.position()
        cos_sin = self.cos[:, t0 : t0 + T], self.sin[:, t0 : t0 + T]

        x = norm(self.transformer.wte(idx).to(self.compute_dtype))
        x = self._smear(x, kv_cache)
        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        x = norm(x - self.backout_lambda.to(x.dtype) * x_backout)

        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)

    def _smear(self, x: torch.Tensor, kv_cache: KVCache | None) -> torch.Tensor:
        """Mix a gated copy of the previous position's embedding into each position."""
        lam = self.smear_lambda.to(x.dtype)
        if kv_cache is None:
            gate = lam * torch.sigmoid(self.smear_gate(x[:, 1:, :SMEAR_GATE_CHANNELS]))
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        previous = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if x.size(1) > 1:
            gate = lam * torch.sigmoid(self.smear_gate(x[:, 1:, :SMEAR_GATE_CHANNELS]))
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        if previous is not None:
            gate = lam * torch.sigmoid(self.smear_gate(x[:, :, :SMEAR_GATE_CHANNELS]))
            return x + gate * previous
        return x
