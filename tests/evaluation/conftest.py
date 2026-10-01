"""A tiny random model with a byte-level tokenizer, loaded the way the evaluation tools load checkpoints."""

from __future__ import annotations

import pytest
import tiktoken
import torch

from wildai.evaluation.runtime import LoadedModel
from wildai.training.model import GPT, GPTConfig
from wildai.training.tokenizer import BOS, Tokenizer

CONFIG = GPTConfig(sequence_len=32, vocab_size=257, n_layer=2, n_head=1, n_kv_head=1, n_embd=128)


@pytest.fixture(scope="session")
def loaded() -> LoadedModel:
    encoding = tiktoken.Encoding(name="bytes", pat_str=r".", mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={BOS: 256})
    torch.manual_seed(0)
    with torch.device("meta"):
        model = GPT(CONFIG)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    tokenizer = Tokenizer(encoding)
    return LoadedModel(model, tokenizer, tokenizer.token_bytes(), torch.device("cpu"), step=0)
