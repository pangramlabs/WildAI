"""GPT-2 token counts, the paper's unit for pool sizes and token-weighted AI shares.

Counts use the GPT-2 byte-level BPE tokenizer at a pinned revision, without special tokens and without truncation. This
is FineWeb's ``token_count`` convention (DataTrove's ``TokensCounter`` with the ``gpt2`` tokenizer and no EOS token).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tokenizers import Tokenizer

GPT2_REPO = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


class TokenCounter(Protocol):
    def count_batch(self, texts: Sequence[str]) -> list[int]: ...


class Gpt2TokenCounter:
    """Counts GPT-2 tokens; the tokenizer file is fetched once from the Hugging Face Hub at the pinned revision."""

    def __init__(self, repo: str = GPT2_REPO, revision: str = GPT2_REVISION) -> None:
        self.repo = repo
        self.revision = revision

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(self.repo, revision=self.revision)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(e.ids) for e in encodings]
