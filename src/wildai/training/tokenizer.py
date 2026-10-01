# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""The 32,768-token BPE tokenizer every WildAI model uses.

It is a tiktoken encoding (byte-level BPE from nanochat's tokenizer trainer). The release ships it as ``tokenizer.json``
(Hugging Face format: at the root of ``pangram/WildAI-models`` and in every model folder), which reads back into the
identical tiktoken encoding; a training run's pickled ``tokenizer.pkl`` loads too. Every document is encoded as
``<|bos|>`` followed by its text tokens.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import tiktoken
import torch

BOS = "<|bos|>"
JSON_FILE = "tokenizer.json"
PICKLE_FILE = "tokenizer.pkl"


def byte_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from bytes to printable characters, used by byte-level BPE vocabularies."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, chars)))


def encoding_from_json(path: Path) -> tiktoken.Encoding:
    """The tiktoken encoding behind a ``tokenizer.json`` written by ``wildai.hub.tokenizer``: vocabulary ids are the
    merge ranks, the first pre-tokenizer holds the split regex, and the added tokens are the special tokens."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    to_byte = {char: byte for byte, char in byte_to_unicode().items()}
    ranks = {bytes(to_byte[c] for c in token): rank for token, rank in spec["model"]["vocab"].items()}
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    specials = {token["content"]: token["id"] for token in spec["added_tokens"]}
    return tiktoken.Encoding(name="wildai", pat_str=pattern, mergeable_ranks=ranks, special_tokens=specials)


class Tokenizer:
    """Thin wrapper around a tiktoken encoding with the BOS convention of the training data."""

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self.encoding = encoding
        self.bos_id = encoding.encode_single_token(BOS)

    @classmethod
    def load(cls, source: Path | None = None) -> Tokenizer:
        """``tokenizer.json`` or ``tokenizer.pkl``, a directory holding one, or by default the released tokenizer."""
        if source is None:
            from wildai.hub.download import tokenizer_file

            source = tokenizer_file()
        path = Path(source)
        if path.is_dir():
            path = next((path / name for name in (JSON_FILE, PICKLE_FILE) if (path / name).exists()), path / JSON_FILE)
        if path.suffix == ".pkl":
            return cls(pickle.loads(path.read_bytes()))  # load only files you trust: unpickling runs code
        return cls(encoding_from_json(path))

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 of the merge ranks, split regex and special tokens: the same for the pickled and the released form."""
        digest = hashlib.sha256(self.encoding._pat_str.encode())
        for table in (self.encoding._mergeable_ranks, {k.encode(): v for k, v in self.encoding._special_tokens.items()}):
            for token, rank in sorted(table.items(), key=lambda item: item[1]):
                digest.update(rank.to_bytes(4, "big") + len(token).to_bytes(4, "big") + token)
        return digest.hexdigest()

    def token_bytes(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Bytes per token id for bits per byte (int64, shape ``(vocab_size,)``), as nanochat counts them: the UTF-8 length
        of the token decoded on its own, so a token holding part of a character counts the 3 bytes of U+FFFD; special
        tokens count 0."""
        special = {self.encoding.encode_single_token(t) for t in self.encoding.special_tokens_set}
        lengths = [0 if i in special else len(self.encoding.decode([i]).encode("utf-8")) for i in range(self.vocab_size)]
        return torch.tensor(lengths, dtype=torch.int64, device=device)

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Text tokens only, without BOS."""
        return self.encoding.encode_ordinary(text)

    def encode_documents(self, texts: Sequence[str], num_threads: int = 8) -> list[list[int]]:
        """Each text as `[BOS, *tokens]`."""
        encoded = self.encoding.encode_ordinary_batch(list(texts), num_threads=num_threads)
        return [[self.bos_id, *ids] for ids in encoded]

    def decode(self, ids: Sequence[int]) -> str:
        return self.encoding.decode(list(ids))

