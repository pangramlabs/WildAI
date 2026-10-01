"""Loading from the Hub layout (one repository, one subfolder per model) through an offline Hub cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import StagedRelease
from .helpers import fake_hub_snapshot, run_isolated

LOAD = """
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args, kwargs = {args!r}, {kwargs!r}
tokenizer = AutoTokenizer.from_pretrained(*args, trust_remote_code=True, **kwargs)
model = AutoModelForCausalLM.from_pretrained(*args, trust_remote_code=True, dtype=torch.float32, **kwargs)
out = model.generate(**tokenizer("The quick brown", return_tensors="pt"), max_new_tokens=5, do_sample=False)
print(json.dumps({{"name": model.config.release["name"], "eos": model.generation_config.eos_token_id, "tokens": out.shape[1]}}))
"""


@pytest.mark.slow
def test_subfolder_of_a_hub_repository(staged_release: StagedRelease, tmp_path: Path) -> None:
    home = tmp_path / "home"
    fake_hub_snapshot(home / "hub", staged_release.repo_id, staged_release.repo_dir)
    for name in staged_release.names:
        result = run_isolated(LOAD.format(args=(staged_release.repo_id,), kwargs={"subfolder": name}), home)
        assert result["name"] == name and result["eos"] == 375


@pytest.mark.slow
def test_remote_code_is_read_from_the_repository_root(staged_release: StagedRelease, tmp_path: Path) -> None:
    """transformers fetches trust_remote_code files from the root even with subfolder=, so the root must carry them."""
    home = tmp_path / "home"
    fake_hub_snapshot(home / "hub", staged_release.repo_id, staged_release.repo_dir, exclude=frozenset({"modeling_wildai.py"}))
    with pytest.raises(RuntimeError, match="OSError"):
        run_isolated(LOAD.format(args=(staged_release.repo_id,), kwargs={"subfolder": staged_release.names[1]}), home)


@pytest.mark.slow
def test_local_copies_load_with_and_without_subfolder(staged_release: StagedRelease, tmp_path: Path) -> None:
    name = staged_release.names[1]
    with_subfolder = run_isolated(LOAD.format(args=(str(staged_release.repo_dir),), kwargs={"subfolder": name}), tmp_path / "a")
    alone = run_isolated(LOAD.format(args=(str(staged_release.repo_dir / name),), kwargs={}), tmp_path / "b")
    assert with_subfolder == alone and alone["name"] == name
