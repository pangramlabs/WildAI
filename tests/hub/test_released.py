"""Reading the release back into the training code: the tokenizer from tokenizer.json and the weights from a model folder."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import tiktoken
import torch

from wildai.evaluation.runtime import load
from wildai.hub.checkpoint import ExportDtype, NanochatCheckpoint
from wildai.hub.tokenizer import load_encoding, save_hf_tokenizer, to_hf_tokenizer
from wildai.training.checkpoint import load_model
from wildai.training.released import load_released
from wildai.training.tokenizer import Tokenizer, encoding_from_json

from .conftest import StagedRelease
from .helpers import TinyCheckpoint


def same_encoding(a: tiktoken.Encoding, b: tiktoken.Encoding) -> bool:
    return a._pat_str == b._pat_str and a._mergeable_ranks == b._mergeable_ranks and a._special_tokens == b._special_tokens


def test_tokenizer_json_reads_back_into_the_same_encoding(encoding: tiktoken.Encoding, tmp_path: Path) -> None:
    save_hf_tokenizer(to_hf_tokenizer(encoding), tmp_path, model_max_length=512)
    assert same_encoding(encoding_from_json(tmp_path / "tokenizer.json"), encoding)
    (tmp_path / "pickled").mkdir()
    (tmp_path / "pickled" / "tokenizer.pkl").write_bytes(pickle.dumps(encoding))
    released, pickled = Tokenizer.load(tmp_path), Tokenizer.load(tmp_path / "pickled")
    assert released.fingerprint == pickled.fingerprint
    assert torch.equal(released.token_bytes(), pickled.token_bytes())


def test_token_bytes(encoding: tiktoken.Encoding) -> None:
    tokenizer = Tokenizer(encoding)
    lengths = tokenizer.token_bytes()
    assert lengths[tokenizer.bos_id] == 0
    ids = tokenizer.encode("hello world")
    assert int(lengths[ids].sum()) == len("hello world")
    assert lengths[encoding.encode_single_token(b"\x80")] == 3  # an incomplete character decodes to U+FFFD


@pytest.mark.skipif("WILDAI_TOKENIZER" not in os.environ, reason="set WILDAI_TOKENIZER to the training tokenizer.pkl")
def test_released_tokenizer_is_the_training_tokenizer(tmp_path: Path) -> None:
    path = Path(os.environ["WILDAI_TOKENIZER"])
    encoding = load_encoding(path)
    save_hf_tokenizer(to_hf_tokenizer(encoding), tmp_path, model_max_length=2048)
    released = Tokenizer.load(tmp_path / "tokenizer.json")
    assert same_encoding(released.encoding, encoding)
    token_bytes = path.parent / "token_bytes.pt"
    if token_bytes.exists():
        assert torch.equal(released.token_bytes(), torch.load(token_bytes, map_location="cpu").long())


def test_released_weights_are_the_training_weights(staged_release: StagedRelease, tiny_checkpoint: TinyCheckpoint) -> None:
    """Every tensor is the training tensor rounded to bfloat16 (the values the bfloat16 forward pass used)."""
    folder = staged_release.repo_dir / staged_release.names[1]
    released = load_released(folder, torch.device("cpu"), compute_dtype=torch.float32).state_dict()
    trained = NanochatCheckpoint.load(tiny_checkpoint.model_path).state_dict
    assert released.keys() == trained.keys()
    for key, tensor in trained.items():
        assert torch.equal(released[key], tensor.to(ExportDtype.bfloat16.torch_dtype).float()), key


def test_released_model_computes_the_training_forward(staged_release: StagedRelease, tiny_checkpoint: TinyCheckpoint) -> None:
    folder = staged_release.repo_dir / staged_release.names[1]
    loaded = load(model=str(folder), device="cpu")
    assert loaded.step == 150 and loaded.tokenizer.vocab_size == loaded.model.config.vocab_size
    reference, _ = load_model(tiny_checkpoint.model_path, torch.device("cpu"), compute_dtype=torch.float32)
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(p.to(torch.bfloat16).float())  # the release stores bfloat16
        ids = torch.tensor([loaded.tokenizer.encode_documents(["The quick brown fox jumps over the lazy dog."])[0]])
        assert torch.allclose(loaded.model(ids), reference(ids), atol=1e-5)
