"""Load a released model (a model folder of ``pangram/WildAI-models``) into the training model class.

A released ``model.safetensors`` holds the training model's tensors under Hugging Face names (``hf_key``), in bfloat16:
the values the bfloat16 forward pass computed with. Loading restores the training layout (float32 matrices and scalars,
embedding tables in the compute dtype), so the training code's forward pass and the evaluation tools run on it unchanged.

    model = load_released(Path("WildAI-models/268m-g072-ai-r0.5"), torch.device("cuda"))
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from wildai.training.model import GPT, GPTConfig, compute_dtype_for

# training state-dict key -> Hugging Face key.
_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), target)
    for pattern, target in (
        (r"transformer\.wte\.weight", "model.embed_tokens.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_q\.weight", r"model.layers.\1.self_attn.q_proj.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_k\.weight", r"model.layers.\1.self_attn.k_proj.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_v\.weight", r"model.layers.\1.self_attn.v_proj.weight"),
        (r"transformer\.h\.(\d+)\.attn\.c_proj\.weight", r"model.layers.\1.self_attn.o_proj.weight"),
        (r"transformer\.h\.(\d+)\.attn\.ve_gate\.weight", r"model.layers.\1.self_attn.value_embed_gate.weight"),
        (r"transformer\.h\.(\d+)\.mlp\.c_fc\.weight", r"model.layers.\1.mlp.up_proj.weight"),
        (r"transformer\.h\.(\d+)\.mlp\.c_proj\.weight", r"model.layers.\1.mlp.down_proj.weight"),
        (r"value_embeds\.(\d+)\.weight", r"model.layers.\1.self_attn.value_embed.weight"),
        (r"resid_lambdas", "model.resid_lambdas"),
        (r"x0_lambdas", "model.x0_lambdas"),
        (r"smear_gate\.weight", "model.smear_gate.weight"),
        (r"smear_lambda", "model.smear_lambda"),
        (r"backout_lambda", "model.backout_lambda"),
        (r"lm_head\.weight", "lm_head.weight"),
    )
)
_EMBEDDINGS = re.compile(r"transformer\.wte\.weight|value_embeds\.\d+\.weight")


def hf_key(training_key: str) -> str:
    for pattern, target in _RENAMES:
        if pattern.fullmatch(training_key):
            return pattern.sub(target, training_key)
    raise KeyError(f"no Hugging Face name for checkpoint key {training_key!r}")


def released_config(folder: Path) -> dict[str, Any]:
    """The folder's ``config.json``; its ``release`` block is the model's row of ``results/models.csv``."""
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def gpt_config(config: dict[str, Any]) -> GPTConfig:
    return GPTConfig(
        sequence_len=config["max_position_embeddings"],
        vocab_size=config["vocab_size"],
        n_layer=config["num_hidden_layers"],
        n_head=config["num_attention_heads"],
        n_kv_head=config["num_key_value_heads"],
        n_embd=config["hidden_size"],
        window_pattern=config["window_pattern"],
    )


def load_released(folder: Path, device: torch.device, compute_dtype: torch.dtype | None = None) -> GPT:
    """The released model in ``folder`` as a training ``GPT`` in eval mode; every tensor must be matched exactly."""
    dtype = compute_dtype or compute_dtype_for(device)
    with torch.device("meta"):
        model = GPT(gpt_config(released_config(folder)))
    stored = load_file(folder / "model.safetensors", device=str(device))
    state = {key: stored.pop(hf_key(key)).to(dtype if _EMBEDDINGS.fullmatch(key) else torch.float32) for key in model.state_dict()}
    if stored:
        raise ValueError(f"{folder} has tensors the model does not: {sorted(stored)[:5]}")
    model.to_empty(device=device)
    model.init_weights(dtype)  # builds the rotary tables; every parameter is then replaced by the released one
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()
