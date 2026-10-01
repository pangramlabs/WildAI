"""Where each released model lives on the Hugging Face Hub.

One repository (``pangram/WildAI-models``) holds every model in a subfolder named by its public name
(``268m-g072-ai-r0.5``). The repository root carries the card that lists every model and the remote-code files, which
``transformers`` always fetches from the root, even with ``subfolder=``; every subfolder also carries its own copy of the
code, card and tokenizer, so a downloaded subfolder loads on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseNaming:
    org: str = "pangram"
    repo: str = "WildAI-models"

    @property
    def repo_id(self) -> str:
        return f"{self.org}/{self.repo}"

    def location(self, name: str) -> ModelLocation:
        return ModelLocation(repo_id=self.repo_id, subfolder=name)

    def staging_dir(self, root: Path) -> Path:
        """Where the repository is staged under ``root``."""
        return root / self.repo


@dataclass(frozen=True)
class ModelLocation:
    repo_id: str
    subfolder: str

    def load_arguments(self) -> str:
        """The positional and keyword arguments of ``from_pretrained`` as Python source."""
        return f'"{self.repo_id}", subfolder="{self.subfolder}"'

    def staging_dir(self, root: Path) -> Path:
        """Where the model is staged under ``root`` (``<root>/<repository>/<model>``)."""
        return root / self.repo_id.split("/", 1)[1] / self.subfolder
