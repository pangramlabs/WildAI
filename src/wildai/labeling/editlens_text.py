"""EditLens input preparation: text cleaning and window selection (no model code, so it is cheap to test).

Cleaning follows the preprocessing EditLens was trained with (``scripts/preprocess.py`` of
https://github.com/pangramlabs/EditLens): emoji to text, drop a reasoning block ending in ``</think>``, drop a first line
that opens like a chatbot preamble ("Sure", "Here", ...), lower-case, and collapse whitespace.

Long documents are scored on up to ``k`` evenly spaced, non-overlapping windows of the tokenized document (first, middle
and last for ``k = 3``); the model's bucket probabilities are averaged over the windows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import emoji

PREAMBLE_STARTS = ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")
_LEADING_SYMBOLS = re.compile(r"^[^a-zA-Z0-9]*")
_WHITESPACE = re.compile(r"\s+")


def _drop_preamble(text: str) -> str:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return text
    first = emoji.replace_emoji(_LEADING_SYMBOLS.sub("", lines[0]), "")
    return "\n".join(lines[1:]) if first.startswith(PREAMBLE_STARTS) else text


def clean_text(text: str) -> str:
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    text = _drop_preamble(text).lower()
    return _WHITESPACE.sub(" ", text).strip()


def select_windows(token_ids: Sequence[int], window: int, k: int) -> list[list[int]]:
    """Up to ``k`` evenly spaced windows of ``window`` tokens from the document's non-overlapping windows."""

    ids = list(token_ids)
    if len(ids) <= window:
        return [ids]
    windows = [ids[start:start + window] for start in range(0, len(ids), window)]
    if len(windows) <= k:
        return windows
    if k == 1:
        return [windows[0]]
    picks = sorted({round(i * (len(windows) - 1) / (k - 1)) for i in range(k)})
    return [windows[p] for p in picks]
