"""The Hugging Face model against the training code's forward pass (``wildai.training``), on the same weights."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wildai.hub.card import CardSettings
from wildai.hub.catalog import Catalog
from wildai.hub.checkpoint import ExportDtype, NanochatCheckpoint
from wildai.hub.export import Exporter
from wildai.hub.tokenizer import load_encoding, to_hf_tokenizer
from wildai.training.checkpoint import load_model

from .helpers import TinyCheckpoint


@pytest.mark.parametrize(("compute", "storage"), [(torch.float32, ExportDtype.float32), (torch.bfloat16, ExportDtype.bfloat16)])
@torch.no_grad()
def test_logits_equal_the_training_forward(tiny_checkpoint: TinyCheckpoint, compute: torch.dtype, storage: ExportDtype) -> None:
    """Bit-identical on CPU, where both use the same SDPA kernels (400 tokens: past the 128-token windows)."""
    reference, _ = load_model(tiny_checkpoint.model_path, torch.device("cpu"), compute_dtype=compute)
    model = NanochatCheckpoint.load(tiny_checkpoint.model_path).to_hf_model(storage, bos_token_id=tiny_checkpoint.vocab_size - 9)
    ids = torch.randint(0, tiny_checkpoint.vocab_size - 9, (2, 400), generator=torch.Generator().manual_seed(0))
    assert torch.equal(model(ids, use_cache=False).logits, reference(ids))


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not {"WILDAI_TEST_MODEL", "WILDAI_TEST_CHECKPOINT", "WILDAI_TOKENIZER"} <= set(os.environ) or not torch.cuda.is_available(),
    reason="set WILDAI_TEST_MODEL (public name), WILDAI_TEST_CHECKPOINT (its model_<step>.pt) and WILDAI_TOKENIZER; needs a GPU",
)
@torch.no_grad()
def test_released_model_reproduces_the_training_loss(tmp_path: Path, sample_texts: list[str]) -> None:
    """Export a real checkpoint, load it back from disk and compare per-token losses on GPU in bfloat16.

    The training code uses Flash Attention 3 on Hopper GPUs and the export uses SDPA, so logits differ by kernel noise.
    """
    name, checkpoint = os.environ["WILDAI_TEST_MODEL"], Path(os.environ["WILDAI_TEST_CHECKPOINT"])
    exporter = Exporter(Catalog.load(), to_hf_tokenizer(load_encoding(Path(os.environ["WILDAI_TOKENIZER"]))), CardSettings())
    folder = exporter.export(name, checkpoint, tmp_path)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, dtype=torch.bfloat16).to(device)
    reference, _ = load_model(checkpoint, device, compute_dtype=torch.bfloat16)
    ids = torch.tensor([[i for text in sample_texts for i in tokenizer(text).input_ids][:2048]], device=device)
    ours = torch.nn.functional.cross_entropy(model(ids, use_cache=False).logits[0, :-1], ids[0, 1:], reduction="none")
    theirs = torch.nn.functional.cross_entropy(reference(ids)[0, :-1], ids[0, 1:], reduction="none")
    assert abs(ours.mean().item() - theirs.mean().item()) < 1e-3
    assert (ours - theirs).abs().median().item() < 1e-2
