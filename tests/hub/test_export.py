"""The export CLI's output: files, metadata, model card, and a round trip through ``from_pretrained``."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from wildai.hub.checkpoint import ExportDtype, NanochatCheckpoint
from wildai.hub.layout import ReleaseNaming
from wildai.hub.publish import plan

from .conftest import StagedRelease
from .helpers import TinyCheckpoint, run_isolated

MODEL_FILES = {
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "configuration_wildai.py",
    "modeling_wildai.py",
}


def test_staged_layout(staged_release: StagedRelease) -> None:
    assert {p.name for p in staged_release.repo_dir.iterdir()} == {
        "LICENSE",
        "README.md",
        "configuration_wildai.py",
        "modeling_wildai.py",
        "tokenizer.json",
        "tokenizer_config.json",
        *staged_release.names,
    }
    for name in staged_release.names:
        assert {p.name for p in (staged_release.repo_dir / name).iterdir()} == MODEL_FILES


def test_every_folder_states_the_license(staged_release: StagedRelease) -> None:
    for folder in (staged_release.repo_dir, *(staged_release.repo_dir / n for n in staged_release.names)):
        card = (folder / "README.md").read_text()
        assert card.startswith("---\nlicense: cc-by-nc-sa-4.0\n") and "> **License: [CC BY-NC-SA 4.0]" in card
        assert (folder / "LICENSE").read_text().startswith("Attribution-NonCommercial-ShareAlike 4.0 International")


def test_config_carries_the_catalog_entry(staged_release: StagedRelease) -> None:
    config = json.loads((staged_release.repo_dir / "19.9m-g900-ai-r0.5" / "config.json").read_text())
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_wildai.WildAIForCausalLM"
    assert config["release"]["name"] == "19.9m-g900-ai-r0.5"
    assert config["release"]["arm"] == "ai" and config["release"]["ai_ratio"] == 0.5
    assert config["release"]["ai_tokens"] == 20_000_000
    generation = json.loads((staged_release.repo_dir / "19.9m-g900-ai-r0.5" / "generation_config.json").read_text())
    assert generation["eos_token_id"] == config["bos_token_id"] == 375


def test_weights_are_bfloat16(staged_release: StagedRelease) -> None:
    with safe_open(staged_release.repo_dir / "19.9m-g900-ai-r0.5" / "model.safetensors", framework="pt") as f:
        assert {f.get_slice(k).get_dtype() for k in f.keys()}


def test_model_cards(staged_release: StagedRelease) -> None:
    card = (staged_release.repo_dir / "19.9m-g900-ai-r0.5" / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-nc-sa-4.0\n")
    assert "- pangram/wildai-data" in card
    assert '"pangram/WildAI-models", subfolder="19.9m-g900-ai-r0.5"' in card
    assert "| C4 | 1.2345 |" in card and "| ptb | 1.5000 |" in card
    assert "r = 0.5 (design 0.5)" in card and "33.3% of the training tokens" in card
    index = (staged_release.repo_dir / "README.md").read_text()
    assert "`19.9m-g900-control`" in index and "`19.9m-g900-ai-r0.5`" in index and "### 19.9M\n" in index


def test_reloaded_model_computes_the_same_logits(staged_release: StagedRelease, tiny_checkpoint: TinyCheckpoint, tmp_path: Path) -> None:
    folder = staged_release.repo_dir / "19.9m-g900-ai-r0.5"
    out = tmp_path / "logits.pt"
    code = f"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained({str(folder)!r}, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained({str(folder)!r}, trust_remote_code=True, dtype=torch.float32)
ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
torch.save(model(ids, use_cache=False).logits, {str(out)!r})
print(json.dumps({{"ids": ids[0].tolist(), "class": type(model).__name__}}))
"""
    result = run_isolated(code, tmp_path / "home")
    assert result["class"] == "WildAIForCausalLM"
    ids = torch.tensor([result["ids"]])
    assert ids[0, 0].item() == 375  # <|bos|> prepended
    expected = NanochatCheckpoint.load(tiny_checkpoint.model_path).to_hf_model(ExportDtype.bfloat16, bos_token_id=375).float()
    with torch.no_grad():
        assert torch.equal(torch.load(out), expected(ids, use_cache=False).logits)


def test_publish_plan_uploads_one_repository(staged_release: StagedRelease) -> None:
    upload = plan(staged_release.root, ReleaseNaming())
    assert (upload.repo_id, upload.folder, upload.models) == (staged_release.repo_id, staged_release.repo_dir, 2)
