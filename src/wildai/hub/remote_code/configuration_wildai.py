"""Configuration of the WildAI language models (a nanochat GPT variant) for Hugging Face ``transformers``.

This file is copied verbatim into every released model repository and loaded with ``trust_remote_code=True``.
It depends only on ``transformers``.
"""

from __future__ import annotations

from typing import ClassVar

from transformers import PretrainedConfig


class WildAIConfig(PretrainedConfig):
    """Architecture of a WildAI model. Only width and depth vary across the released models.

    Every layer uses causal attention limited to a left window: a query attends to itself and the ``window`` tokens before
    it. ``window_pattern`` is tiled over the layers ("S" = a quarter of the context rounded up to a multiple of 128 tokens,
    "L" = the full context) and the last layer always uses "L".

    ``release`` holds the model's entry in the release catalog (name, token counts, AI ratio, ...); it does not affect
    the forward pass.
    """

    model_type = "wildai"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        max_position_embeddings: int = 2048,
        window_pattern: str = "SSSL",
        rotary_base: float = 100000.0,
        qk_scale: float = 1.2,
        logit_softcap: float = 15.0,
        value_embed_gate_channels: int = 12,
        value_embed_gate_range: float = 3.0,
        smear_channels: int = 24,
        bos_token_id: int = 32759,
        eos_token_id: int = 32759,
        use_cache: bool = True,
        release: dict[str, str | int | float | None] | None = None,
        **kwargs: object,
    ) -> None:
        if hidden_size % num_attention_heads != 0 or num_attention_heads % num_key_value_heads != 0:
            raise ValueError("hidden_size must divide into num_attention_heads, and those into num_key_value_heads")
        if not window_pattern or set(window_pattern.upper()) - {"S", "L"}:
            raise ValueError(f"window_pattern must be a non-empty string of S and L, got {window_pattern!r}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.window_pattern = window_pattern.upper()
        self.rotary_base = rotary_base
        self.qk_scale = qk_scale
        self.logit_softcap = logit_softcap
        self.value_embed_gate_channels = value_embed_gate_channels
        self.value_embed_gate_range = value_embed_gate_range
        self.smear_channels = smear_channels
        self.use_cache = use_cache
        self.release = release
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        return 4 * self.hidden_size

    @property
    def window_sizes(self) -> list[int]:
        """Left attention window of each layer, in tokens."""
        long_window = self.max_position_embeddings
        short_window = -(-long_window // 4 // 128) * 128  # a quarter of the context, rounded up to a multiple of 128 (2048 -> 512)
        pattern = self.window_pattern
        sizes = [short_window if pattern[i % len(pattern)] == "S" else long_window for i in range(self.num_hidden_layers)]
        sizes[-1] = long_window
        return sizes

    def has_value_embedding(self, layer_idx: int) -> bool:
        """Value embeddings sit on alternating layers, always including the last one."""
        return layer_idx % 2 == (self.num_hidden_layers - 1) % 2

    @property
    def backout_layer(self) -> int:
        """The residual stream after this layer is partly subtracted before the final norm."""
        return self.num_hidden_layers // 2
