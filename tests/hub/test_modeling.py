"""The Hugging Face model: attention windows, cache and padding consistency, generation, and the checkpoint conversion."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from transformers.cache_utils import DynamicCache

from wildai.hub.checkpoint import ExportDtype, NanochatCheckpoint, NanochatModelConfig
from wildai.hub.remote_code.configuration_wildai import WildAIConfig
from wildai.hub.remote_code.modeling_wildai import WildAIForCausalLM, window_masks
from wildai.training.released import hf_key

from .helpers import TinyCheckpoint


@pytest.fixture(scope="module")
def model(tiny_checkpoint: TinyCheckpoint) -> WildAIForCausalLM:
    return NanochatCheckpoint.load(tiny_checkpoint.model_path).to_hf_model(ExportDtype.float32, bos_token_id=bos_id(tiny_checkpoint))


def bos_id(checkpoint: TinyCheckpoint) -> int:
    return checkpoint.vocab_size - 9  # <|bos|> is the first of the nine special tokens


def tokens(length: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
    ids = torch.randint(0, vocab_size - 9, (1, length), generator=torch.Generator().manual_seed(seed))
    ids[:, 0] = vocab_size - 9
    return ids


def test_window_sizes_follow_the_training_rule() -> None:
    config = WildAIConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2)
    assert config.window_sizes == [512, 512, 512, 2048, 512, 2048]
    assert [config.has_value_embedding(i) for i in range(6)] == [False, True, False, True, False, True]


def test_window_mask_limits_how_far_back_a_query_sees() -> None:
    masks = window_masks([128, 512], query_len=300, past_len=0, padding_mask=None, device=torch.device("cpu"))
    q = torch.arange(300)[:, None]
    k = torch.arange(300)[None, :]
    assert torch.equal(masks[128].mask, (k <= q) & (q - k <= 128))
    assert masks[512].mask is None and masks[512].is_causal
    decode = window_masks([128], query_len=1, past_len=299, padding_mask=None, device=torch.device("cpu"))[128].mask
    assert decode is not None and int(decode.sum()) == 129


def test_every_checkpoint_key_has_a_home(tiny_checkpoint: TinyCheckpoint) -> None:
    checkpoint = NanochatCheckpoint.load(tiny_checkpoint.model_path)
    assert {hf_key(k) for k in checkpoint.state_dict} == set(
        checkpoint.to_hf_model(ExportDtype.bfloat16, bos_id(tiny_checkpoint)).state_dict()
    )
    with pytest.raises(KeyError):
        hf_key("transformer.h.0.attn.c_q.bias")


def test_only_the_released_architecture_is_accepted() -> None:
    base = {"sequence_len": 2048, "vocab_size": 32768, "n_layer": 4, "n_head": 2, "n_kv_head": 2, "n_embd": 256, "window_pattern": "SSSL"}
    NanochatModelConfig(**base)
    for switch in ("smear", "backout", "value_embeds", "qk_norm", "layer_lambdas", "logit_softcap", "embed_norm"):
        with pytest.raises(ValidationError):
            NanochatModelConfig(**base, **{switch: False})
    with pytest.raises(ValidationError):
        NanochatModelConfig(**base, mlp_act="gelu")


def test_bfloat16_export_holds_the_values_training_computed_with(tiny_checkpoint: TinyCheckpoint) -> None:
    checkpoint = NanochatCheckpoint.load(tiny_checkpoint.model_path)
    exported = checkpoint.hf_state_dict(ExportDtype.bfloat16)
    original = checkpoint.state_dict["transformer.h.0.attn.c_q.weight"]
    assert torch.equal(exported["model.layers.0.self_attn.q_proj.weight"], original.to(torch.bfloat16))
    assert {t.dtype for t in exported.values()} == {torch.bfloat16}


@torch.no_grad()
def test_cached_decoding_matches_the_full_forward(model: WildAIForCausalLM) -> None:
    ids = tokens(400, model.config.vocab_size)  # longer than the 128-token windows
    full = model(ids, use_cache=False).logits
    cache = DynamicCache()
    pieces = [model(ids[:, :300], past_key_values=cache).logits]
    pieces.append(model(ids[:, 300:350], past_key_values=cache, previous_token_ids=ids[:, 299:300]).logits)
    for t in range(350, 400):
        pieces.append(model(ids[:, t : t + 1], past_key_values=cache, previous_token_ids=ids[:, t - 1 : t]).logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="previous_token_ids"):
        model(ids[:, :1], past_key_values=cache)


@torch.no_grad()
def test_left_padding_does_not_change_the_result(model: WildAIForCausalLM) -> None:
    ids = tokens(50, model.config.vocab_size, seed=1)
    padded = torch.cat([torch.full((1, 7), model.config.bos_token_id), ids], dim=1)
    mask = torch.cat([torch.zeros(1, 7, dtype=torch.long), torch.ones(1, 50, dtype=torch.long)], dim=1)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    got = model(padded, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, 7:]
    torch.testing.assert_close(got, model(ids, use_cache=False).logits, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_generation_with_and_without_cache_agree(model: WildAIForCausalLM) -> None:
    ids = tokens(20, model.config.vocab_size, seed=2)
    kwargs = {"max_new_tokens": 30, "do_sample": False, "eos_token_id": None}
    with_cache = model.generate(ids, **kwargs)
    without_cache = model.generate(ids, use_cache=False, **kwargs)
    assert with_cache.shape == (1, 50)
    assert torch.equal(with_cache, without_cache)
    beams = model.generate(ids, num_beams=3, **kwargs)
    assert beams.shape == (1, 50)


@torch.no_grad()
def test_loss_is_the_shifted_cross_entropy(model: WildAIForCausalLM) -> None:
    ids = tokens(64, model.config.vocab_size, seed=3)
    out = model(ids, labels=ids, use_cache=False)
    expected = torch.nn.functional.cross_entropy(out.logits[0, :-1], ids[0, 1:])
    torch.testing.assert_close(out.loss, expected)
    assert out.logits.abs().max() <= model.config.logit_softcap
