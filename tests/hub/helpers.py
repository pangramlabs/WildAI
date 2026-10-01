"""Tiny stand-ins for the release inputs: a small tokenizer trained like the real one and random checkpoints in the training format."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import rustbpe
import tiktoken
import torch

# The training tokenizer's split pattern (GPT-4 style, numbers split into runs of at most two digits).
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)

EDGE_CASES = (
    "",
    " ",
    "\n\n\n",
    "\r\n\r\n",
    "\t\t x",
    "a  b   c    d",
    "trailing spaces   ",
    "12345678901234567890 3.14159 2,000,000",
    "don't I'll we've they're SHE'S",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "emoji 😀👍🏽👨‍👩‍👧‍👦 flags 🇺🇸",
    "combining é ä, RTL עברית العربية",
    "def f(x):\n    return x**2  # comment\n",
    "text <|bos|> more <|assistant_start|>hi<|assistant_end|>",
    "<|bos",
    "\x00\x01 control \x7f",
    "a" * 3000,
    " " * 500 + "x",
)

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "Language models are trained on web text; some of it is written by other language models.",
    "In 2026, 27.5% of tokens were AI-generated. Numbers like 1234 and 56 split into pairs.",
    "def train(model, data):\n    for batch in data:\n        loss = model(batch)\n",
    "Résumé, naïve café. Ελληνικά. Русский текст. 日本語。",
] * 20


def train_encoding(vocab_size: int = 384) -> tiktoken.Encoding:
    """A small tokenizer built the way the training tokenizer was: rustbpe merges wrapped in a tiktoken Encoding.

    The vocabulary is a multiple of 64 so the training model does not pad its embedding tables.
    """
    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(iter(CORPUS), vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    ranks = {bytes(k): v for k, v in trainer.get_mergeable_ranks()}
    specials = {name: len(ranks) + i for i, name in enumerate(SPECIAL_TOKENS)}
    return tiktoken.Encoding(name="tiny", pat_str=trainer.get_pattern(), mergeable_ranks=ranks, special_tokens=specials)


@dataclass(frozen=True)
class TinyCheckpoint:
    model_path: Path
    depth: int
    width: int
    vocab_size: int


def nanochat_state_dict(depth: int, width: int, n_head: int, vocab_size: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights with the training code's names, shapes and storage dtypes (bf16 embeddings, fp32 elsewhere)."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    sd = {"transformer.wte.weight": rand(vocab_size, width, scale=1.0, dtype=torch.bfloat16), "lm_head.weight": rand(vocab_size, width)}
    for i in range(depth):
        for name in ("c_q", "c_k", "c_v", "c_proj"):
            sd[f"transformer.h.{i}.attn.{name}.weight"] = rand(width, width)
        sd[f"transformer.h.{i}.mlp.c_fc.weight"] = rand(4 * width, width)
        sd[f"transformer.h.{i}.mlp.c_proj.weight"] = rand(width, 4 * width)
        if i % 2 == (depth - 1) % 2:
            sd[f"transformer.h.{i}.attn.ve_gate.weight"] = rand(n_head, 12)
            sd[f"value_embeds.{i}.weight"] = rand(vocab_size, width, scale=0.5, dtype=torch.bfloat16)
    sd["resid_lambdas"] = 1.0 + rand(depth)
    sd["x0_lambdas"] = 0.1 + rand(depth)
    sd["smear_gate.weight"] = rand(1, 24)
    sd["smear_lambda"] = 0.3 + rand(1)
    sd["backout_lambda"] = 0.2 + rand(1)
    return sd


def write_checkpoint(directory: Path, depth: int, width: int, n_head: int, vocab_size: int, step: int = 42) -> TinyCheckpoint:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"model_{step:06d}.pt"
    torch.save(nanochat_state_dict(depth, width, n_head, vocab_size), model_path)
    model_config = {
        "sequence_len": 512,
        "vocab_size": vocab_size,
        "n_layer": depth,
        "n_head": n_head,
        "n_kv_head": n_head,
        "n_embd": width,
        "window_pattern": "SSSL",
    }
    (directory / f"meta_{step:06d}.json").write_text(json.dumps({"step": step, "model_config": model_config}))
    return TinyCheckpoint(model_path=model_path, depth=depth, width=width, vocab_size=vocab_size)


def fake_hub_snapshot(hub_cache: Path, repo_id: str, source: Path, exclude: frozenset[str] = frozenset()) -> None:
    """Lay out ``source`` (a staged repository) as the ``main`` revision of ``repo_id`` in an offline Hub cache.

    ``huggingface_hub`` resolves ``refs/main`` to a commit and reads files from ``snapshots/<commit>/``, so loading with
    ``HF_HUB_OFFLINE=1`` exercises the same path handling (subfolders, remote code) as a download.
    """
    repo_dir = hub_cache / ("models--" + repo_id.replace("/", "--"))
    commit = hashlib.sha1(repo_id.encode()).hexdigest()
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_file() and relative.as_posix() not in exclude:
            target = repo_dir / "snapshots" / commit / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path.resolve())


def run_isolated(code: str, home: Path) -> dict[str, object]:
    """Run Python ``code`` in a fresh process with its own offline Hugging Face home; it must print one JSON line last."""
    env = {
        **os.environ,
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(home / "hub"),
        "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TRANSFORMERS_VERBOSITY": "error",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])
