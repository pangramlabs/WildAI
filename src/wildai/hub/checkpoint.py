"""Read nanochat training checkpoints and convert them to the Hugging Face model in ``wildai.hub.remote_code``.

A checkpoint is ``model_<step>.pt`` (a plain state dict) next to ``meta_<step>.json`` (its ``model_config``).
Matrices are stored as float32 master weights and embeddings as bfloat16; training and evaluation computed in bfloat16,
casting every matrix to bfloat16 at each matmul, so a bfloat16 export computes the same function (see ``ExportDtype``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from wildai.hub.remote_code.configuration_wildai import WildAIConfig
from wildai.hub.remote_code.modeling_wildai import WildAIForCausalLM
from wildai.training.released import hf_key


class NanochatModelConfig(BaseModel):
    """The ``model_config`` block of a checkpoint's meta file. The switches pin the one architecture the export supports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str
    mlp_act: Literal["relu2"] = "relu2"
    qk_norm: Literal[True] = True
    value_embeds: Literal[True] = True
    smear: Literal[True] = True
    backout: Literal[True] = True
    layer_lambdas: Literal[True] = True
    logit_softcap: Literal[True] = True
    embed_norm: Literal[True] = True
    init_style: str = "nanochat"  # initialisation only; does not change the forward pass

    def to_hf(self, bos_token_id: int) -> WildAIConfig:
        """``bos_token_id`` (<|bos|> in the tokenizer) also ends generation: training documents are delimited by it alone."""
        return WildAIConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.n_embd,
            num_hidden_layers=self.n_layer,
            num_attention_heads=self.n_head,
            num_key_value_heads=self.n_kv_head,
            max_position_embeddings=self.sequence_len,
            window_pattern=self.window_pattern,
            bos_token_id=bos_token_id,
            eos_token_id=bos_token_id,
            pad_token_id=bos_token_id,
        )


class ExportDtype(str, Enum):
    """Storage precision of an export.

    ``bfloat16`` holds exactly the values the bfloat16 forward pass used (the training code cast every weight to
    bfloat16 at use); ``float32`` keeps the training master weights of the matrices and scalars.
    """

    bfloat16 = "bfloat16"
    float32 = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self is ExportDtype.bfloat16 else torch.float32


@dataclass(frozen=True)
class NanochatCheckpoint:
    config: NanochatModelConfig
    state_dict: dict[str, torch.Tensor]
    step: int

    @classmethod
    def load(cls, model_path: Path) -> NanochatCheckpoint:
        """Load ``model_<step>.pt`` and the ``meta_<step>.json`` beside it."""
        match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
        if match is None:
            raise ValueError(f"expected a file named model_<step>.pt, got {model_path.name}")
        step = int(match.group(1))
        meta = json.loads((model_path.parent / f"meta_{step:06d}.json").read_text())
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        return cls(config=NanochatModelConfig(**meta["model_config"]), state_dict=state_dict, step=step)

    def hf_state_dict(self, dtype: ExportDtype) -> dict[str, torch.Tensor]:
        """Renamed tensors in the storage precision of ``dtype``."""
        return {hf_key(key): tensor.to(dtype.torch_dtype).contiguous() for key, tensor in self.state_dict.items()}

    def to_hf_model(self, dtype: ExportDtype, bos_token_id: int) -> WildAIForCausalLM:
        """The Hugging Face model holding this checkpoint's weights; every parameter must be matched exactly."""
        config = self.config.to_hf(bos_token_id)
        config.dtype = dtype.value
        with torch.device("meta"):
            model = WildAIForCausalLM(config)
        model.load_state_dict(self.hf_state_dict(dtype), strict=True, assign=True)
        return model.eval()
