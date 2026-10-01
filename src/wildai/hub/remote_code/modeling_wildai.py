"""The WildAI language models (a nanochat GPT variant) for Hugging Face ``transformers``.

This file is copied verbatim into every released model repository and loaded with ``trust_remote_code=True``.
It depends only on ``torch`` and ``transformers`` and reproduces the training code's forward pass operation by
operation, so a model loaded in bfloat16 computes the same function the paper evaluated:

- token embedding followed by a parameter-free RMSNorm, then a learned "smear" that mixes each position's embedding
  with the previous token's;
- per-layer scalars that rescale the residual stream and blend the initial embedding back in before every block;
- attention with rotary embeddings applied before a per-head RMSNorm of queries and keys (both scaled by 1.2),
  a left sliding window per layer, and gated value embeddings added to the values on alternating layers;
- a squared-ReLU MLP, parameter-free pre-norms, no biases;
- "backout": part of the residual stream after the middle layer is subtracted before the final norm;
- untied output head with logits soft-capped by ``15 * tanh(logits / 15)``.

Attention uses PyTorch's scaled_dot_product_attention with explicit window masks, so no FlashAttention install is
needed. Generation uses the standard ``DynamicCache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_wildai import WildAIConfig


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free RMSNorm over the last dimension with PyTorch's default epsilon, as in training."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (i, i + d/2) of ``x`` (batch, time, heads, head_dim), with the training code's sign convention."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


@dataclass(frozen=True)
class WindowMask:
    """Arguments for scaled_dot_product_attention that implement one attention window."""

    mask: torch.Tensor | None
    is_causal: bool


def window_masks(
    window_sizes: list[int], query_len: int, past_len: int, padding_mask: torch.Tensor | None, device: torch.device
) -> dict[int, WindowMask]:
    """Masks for each distinct left window. Query i (absolute index past_len + i) sees keys j with i - window <= j <= i.

    ``padding_mask`` is the (batch, past_len + query_len) attention mask when it contains padding, else None. A padding
    query is always allowed to see itself so that its (unused) output stays finite.
    """
    total = past_len + query_len
    q_idx = torch.arange(past_len, total, device=device)[:, None]
    k_idx = torch.arange(total, device=device)[None, :]
    masks: dict[int, WindowMask] = {}
    for window in sorted(set(window_sizes)):
        if padding_mask is None and window >= total - 1:
            if query_len == 1:
                masks[window] = WindowMask(mask=None, is_causal=False)
                continue
            if past_len == 0:
                masks[window] = WindowMask(mask=None, is_causal=True)
                continue
        allowed = (k_idx <= q_idx) & (q_idx - k_idx <= window)
        if padding_mask is not None:
            allowed = (allowed & padding_mask[:, None, None, :].bool()) | (k_idx == q_idx)
        masks[window] = WindowMask(mask=allowed, is_causal=False)
    return masks


class WildAIAttention(nn.Module):
    def __init__(self, config: WildAIConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qk_scale = config.qk_scale
        self.window = config.window_sizes[layer_idx]
        hidden, kv_dim = config.hidden_size, config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.value_embed: nn.Embedding | None = None
        self.value_embed_gate: nn.Linear | None = None
        if config.has_value_embedding(layer_idx):
            self.value_embed = nn.Embedding(config.vocab_size, kv_dim)
            self.value_embed_gate = nn.Linear(config.value_embed_gate_channels, config.num_key_value_heads, bias=False)
        self.gate_channels = config.value_embed_gate_channels
        self.gate_range = config.value_embed_gate_range

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.LongTensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mask: WindowMask,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, time, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, time, self.num_kv_heads, self.head_dim)
        if self.value_embed is not None:
            ve = self.value_embed(input_ids).to(x.dtype).view(batch, time, self.num_kv_heads, self.head_dim)
            gate = self.gate_range * torch.sigmoid(self.value_embed_gate(x[..., : self.gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = rotary
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q) * self.qk_scale, rms_norm(k) * self.qk_scale
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.mask, is_causal=mask.is_causal, enable_gqa=self.num_kv_heads != self.num_heads
        )
        return self.o_proj(y.transpose(1, 2).reshape(batch, time, -1))


class WildAIMLP(nn.Module):
    def __init__(self, config: WildAIConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


class WildAIDecoderLayer(nn.Module):
    def __init__(self, config: WildAIConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = WildAIAttention(config, layer_idx)
        self.mlp = WildAIMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.LongTensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mask: WindowMask,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(rms_norm(x), input_ids, rotary, mask, past_key_values)
        return x + self.mlp(rms_norm(x))


class WildAIPreTrainedModel(PreTrainedModel):
    config_class = WildAIConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules: ClassVar[list[str]] = ["WildAIDecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # Released models are always loaded from trained weights; this only gives from_config() models sane values.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, WildAIModel):
            with torch.no_grad():
                module.resid_lambdas.fill_(1.0)
                module.x0_lambdas.zero_()
                module.smear_lambda.zero_()
                module.backout_lambda.zero_()


class WildAIModel(WildAIPreTrainedModel):
    """The transformer trunk; returns the final normalised hidden states."""

    def __init__(self, config: WildAIConfig) -> None:
        super().__init__(config)
        n_layer = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([WildAIDecoderLayer(config, i) for i in range(n_layer)])
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        self.smear_gate = nn.Linear(config.smear_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.zeros(1))
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def rotary(self, position_ids: torch.LongTensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables of shape (batch, time, 1, head_dim / 2), computed in float32 and cast like in training."""
        head_dim = self.config.head_dim
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device)
        inv_freq = 1.0 / (self.config.rotary_base ** (channels / head_dim))
        freqs = position_ids.to(torch.float32)[..., None] * inv_freq
        return freqs.cos().to(dtype)[:, :, None, :], freqs.sin().to(dtype)[:, :, None, :]

    def smear(self, x: torch.Tensor, previous: torch.Tensor | None, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Add a gated copy of the previous position's embedding to every position that has one.

        ``previous`` is the normalised embedding of the token just before this chunk (None at the start of a sequence);
        with left padding, the first real token has a padding token before it and gets nothing.
        """
        channels = self.config.smear_channels
        lam = self.smear_lambda.to(x.dtype)
        first = x[:, :1]
        if previous is not None:
            first = first + lam * torch.sigmoid(self.smear_gate(first[..., :channels])) * previous
        if x.size(1) == 1:
            return first
        gate = lam * torch.sigmoid(self.smear_gate(x[:, 1:, :channels]))
        if padding_mask is not None:
            gate = gate * padding_mask[:, -x.size(1) : -1, None].to(gate.dtype)
        return torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        previous_token_ids: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: object,
    ) -> BaseModelOutputWithPast:
        """``previous_token_ids`` (batch, 1): the token before ``input_ids[:, 0]``; required when continuing from a cache."""
        if input_ids is None:
            raise ValueError("input_ids are required: the value embeddings are looked up by token id")
        batch, time = input_ids.shape
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_len > 0 and previous_token_ids is None:
            raise ValueError("previous_token_ids is required when continuing from past_key_values (the smear needs it)")
        padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, past_len + time):
                raise ValueError(f"attention_mask must have shape {(batch, past_len + time)}, got {tuple(attention_mask.shape)}")
            if not bool(attention_mask.all()):
                padding_mask = attention_mask
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + time, device=input_ids.device).expand(batch, -1)

        x = rms_norm(self.embed_tokens(input_ids))
        previous = rms_norm(self.embed_tokens(previous_token_ids)) if previous_token_ids is not None else None
        x = self.smear(x, previous, padding_mask)
        rotary = self.rotary(position_ids, x.dtype)
        masks = window_masks(self.config.window_sizes, time, past_len, padding_mask, input_ids.device)

        x0 = x
        backout = None
        hidden_states: tuple[torch.Tensor, ...] = (x,)
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = layer(x, input_ids, rotary, masks[layer.self_attn.window], past_key_values)
            if i == self.config.backout_layer:
                backout = x
            hidden_states += (x,)
        x = rms_norm(x - self.backout_lambda.to(x.dtype) * backout)
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class WildAIForCausalLM(WildAIPreTrainedModel, GenerationMixin):
    def __init__(self, config: WildAIConfig) -> None:
        super().__init__(config)
        self.model = WildAIModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        previous_token_ids: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int = 0,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        """Float32 logits after the soft cap. ``labels`` are shifted inside (standard causal-LM loss, -100 ignored)."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            previous_token_ids=previous_token_ids,
            output_hidden_states=output_hidden_states,
        )
        hidden = outputs.last_hidden_state[:, -logits_to_keep:] if logits_to_keep else outputs.last_hidden_state
        logits = self.lm_head(hidden).float()
        cap = self.config.logit_softcap
        logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten().to(logits.device), ignore_index=-100)
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states
        )
        return result.to_tuple() if return_dict is False else result

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs: object) -> dict[str, object]:
        # The smear mixes in the embedding of the token before the first new one; take it from the full sequence.
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        new_len = model_inputs["input_ids"].shape[1]
        if input_ids.shape[1] > new_len:
            model_inputs["previous_token_ids"] = input_ids[:, -new_len - 1 : -new_len]
        return model_inputs
