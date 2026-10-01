"""The model on CPU: shapes, KV-cache consistency, checkpoint round trip, recipe checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wildai.training.checkpoint import load_model, save_checkpoint
from wildai.training.model import GPT, GPTConfig, KVCache

CONFIG = GPTConfig(sequence_len=64, vocab_size=257, n_layer=3, n_head=1, n_kv_head=1, n_embd=128, window_pattern="SSSL")


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = GPT(CONFIG)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    return model


def test_forward_and_loss() -> None:
    model = tiny_model()
    ids = torch.randint(0, 257, (2, 64))
    assert model(ids).shape == (2, 64, 257)
    loss = model(ids[:, :-1], ids[:, 1:])
    assert torch.isfinite(loss) and 4.0 < loss.item() < 7.0  # about log(257) at initialisation


def test_kv_cache_matches_full_forward() -> None:
    model = tiny_model()
    with torch.no_grad():  # train a moment so the smear and value embeddings matter
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ids = torch.randint(0, 257, (2, 20))
    full = model(ids)
    cache = KVCache(CONFIG, 2, 40, torch.device("cpu"), torch.float32)
    steps = [model(ids[:, :12], kv_cache=cache)[:, -1]]
    steps += [model(ids[:, t : t + 1], kv_cache=cache)[:, -1] for t in range(12, 20)]
    torch.testing.assert_close(torch.stack(steps[1:], 1), full[:, 12:20], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(steps[0], full[:, 11], atol=1e-4, rtol=1e-4)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = tiny_model()
    save_checkpoint(tmp_path / "run", 7, model, {"note": "test"})
    loaded, meta = load_model(tmp_path / "run", torch.device("cpu"))
    assert meta["step"] == 7 and meta["model_config"]["n_layer"] == 3
    ids = torch.randint(0, 257, (1, 16))
    torch.testing.assert_close(loaded(ids), model(ids))


def test_param_counts() -> None:
    counts = tiny_model().param_counts()
    assert counts.total == sum(p.numel() for p in tiny_model().parameters())
    assert counts.paper_n == counts.total - counts.value_embeds


def test_config_rejects_other_recipes() -> None:
    assert GPTConfig.from_dict({**CONFIG.to_dict(), "mlp_act": "relu2", "qk_norm": True}) == CONFIG
    with pytest.raises(ValueError, match="qk_norm"):
        GPTConfig.from_dict({**CONFIG.to_dict(), "qk_norm": False})
