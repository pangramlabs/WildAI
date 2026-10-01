"""Which documents a run trains on, and in what order (the paper's additive design and its filtering pairs).

Every run draws from two tokenized pools, one of human-labelled and one of AI-labelled documents, each in its fixed
key order (`wildai.training.pools`). A run is a `RunSpec`: a depth, an arm, its human budget and total length in
optimizer steps, and a seed. A step packs `rows_per_step` rows of `sequence_len + 1` document tokens (the model
predicts every token of a row but the first), so it consumes `step_tokens = rows_per_step * (sequence_len + 1)`
document tokens. As in nanochat's BOS-aligned loader, a document contributes at most one row: its first
`sequence_len + 1` tokens. Budgets count these tokens, and training packs them without discarding any
(`packing.BestFitPacker(keep_remainder=True)`), so a run trains on every one of them exactly once.

A run's human documents are always the first `D_H = human_steps * step_tokens` tokens of the human pool (the last
document is cut to fit exactly). The arms differ in what comes on top of them, `(steps - human_steps) * step_tokens`
tokens:
- control: nothing (`steps == human_steps`).
- ai: the first tokens of the AI pool, so every AI run of a group trains on exactly its control's human documents and
  r = steps / human_steps - 1.
- human: the human documents that follow the control's in the pool's order (fresh human text).
- repeat: the control's documents again, in whole passes and then part of one more, until the run has `steps` steps
  (the step count of the AI run it is matched to).
- natural: the web-mix model of a filtering pair. Like `ai`, but its total length is the budget and the AI pool
  supplies the measured 2026 share of its tokens (`WEB_AI_SHARE_2026`; the paper rate-matches a mixture of the same
  pools rather than training on a crawl sample).
- filtered: the same web mix with its AI documents removed and nothing added: exactly the natural run's human
  documents, trained for correspondingly fewer steps.

Order: documents are sorted by a keyed hash of their pool key and the seed, so every run of a group sees the
control's human documents in the same relative order, with added documents interleaved uniformly. A repeat run's
passes are concatenated, each in its own hash order. The order depends only on the documents and the seed, not on the
number of GPUs. The seed also sets the weight initialization; it does not change which documents are selected.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wildai.training.packing import Piece
from wildai.training.pools import PoolStore
from wildai.training.recipe import Recipe

WEB_AI_SHARE_2026 = 0.22315880974511895
"""AI-labelled share of the tokens of the held-out 2026 web sample (FW26), the web mix of the filtering pairs."""


class Arm(str, Enum):
    CONTROL = "control"
    AI = "ai"
    HUMAN = "human"
    REPEAT = "repeat"
    NATURAL = "natural"
    FILTERED = "filtered"


HUMAN_ONLY_ARMS = (Arm.CONTROL, Arm.FILTERED)


class RunSpec(BaseModel):
    """One training run: `human_steps` steps' worth of human tokens, `steps` optimizer steps in total."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    depth: int
    arm: Arm
    human_steps: int
    steps: int
    seed: int = 1337

    @model_validator(mode="after")
    def _check_steps(self) -> RunSpec:
        if self.human_steps <= 0:
            raise ValueError("a run needs human text")
        if self.arm in HUMAN_ONLY_ARMS and self.steps != self.human_steps:
            raise ValueError(f"a {self.arm.value} run trains on human text only, so steps == human_steps")
        if self.arm not in HUMAN_ONLY_ARMS and self.steps <= self.human_steps:
            raise ValueError(f"a {self.arm.value} run adds tokens to its human text, so steps > human_steps")
        return self

    @classmethod
    def from_budget(
        cls,
        depth: int,
        arm: Arm,
        tpp: float,
        seed: int,
        recipe: Recipe,
        ratio: float = 0.0,
        ai_share: float = WEB_AI_SHARE_2026,
        name: str | None = None,
    ) -> RunSpec:
        """A new run from tokens per parameter (the paper's N) and, for additive arms, the added ratio r.

        Additive arms: `tpp` counts human tokens and the run adds `ratio` tokens per human token (repeat: the
        matched AI run's r). Filtering arms: `tpp` counts all tokens of the web mix, `ai_share` of which are AI text;
        the filtered run keeps only the web mix's human part.
        """
        steps = _round(tpp * recipe.param_counts(depth).paper_n / recipe.batch_tokens(depth))
        if arm in (Arm.NATURAL, Arm.FILTERED):
            human = _round((1 - ai_share) * steps)
            total = steps if arm == Arm.NATURAL else human
            default = f"d{depth}-tpp{tpp:g}-{arm.value}-share{ai_share:.3g}-s{seed}"
        else:
            human = steps
            total = steps if arm == Arm.CONTROL else _round((1 + ratio) * steps)
            default = f"d{depth}-tpp{tpp:g}-{arm.value}" + (f"-r{ratio:g}" if arm != Arm.CONTROL else "") + f"-s{seed}"
        return cls(name=name or default, depth=depth, arm=arm, human_steps=human, steps=total, seed=seed)


def _round(x: float) -> int:
    return int(np.floor(x + 0.5))


POOL_NAMES = ("human", "ai")
HUMAN, AI = 0, 1


@dataclass(frozen=True)
class Pools:
    """The tokenized pools runs draw from (`<data_dir>/human` and `<data_dir>/ai`)."""

    human: PoolStore
    ai: PoolStore | None = None

    @classmethod
    def from_directory(cls, data_dir: Path) -> Pools:
        return cls(PoolStore(data_dir / "human"), PoolStore(data_dir / "ai") if (data_dir / "ai").is_dir() else None)

    def by_id(self, pool_id: int) -> PoolStore:
        store = (self.human, self.ai)[pool_id]
        if store is None:
            raise FileNotFoundError(f"missing pool {POOL_NAMES[pool_id]!r}")
        return store


class PlanSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arm: Arm
    seed: int
    steps: int
    human_steps: int
    step_tokens: int
    trained_tokens: int  # steps * rows_per_step * sequence_len, the paper's token count
    human_documents: int
    human_tokens: int  # document tokens (repeat runs: counting every pass)
    ai_documents: int
    ai_tokens: int
    ai_token_share: float
    passes: float  # document tokens / unique document tokens (1 unless repeated)


@dataclass(frozen=True)
class DocumentPlan:
    """A run's documents in training order: pool id, row in the pool's key order, and tokens used."""

    spec: RunSpec
    step_tokens: int
    pool: np.ndarray
    row: np.ndarray
    length: np.ndarray

    @property
    def steps(self) -> int:
        return self.spec.steps

    @property
    def ai(self) -> np.ndarray:
        return self.pool == AI

    def pieces(self, pools: Pools, batch_size: int = 128) -> Iterator[list[Piece]]:
        """The documents as packer input (tag 1 marks AI text), a batch at a time."""
        for start in range(0, len(self.row), batch_size):
            stop = start + batch_size
            yield [Piece(pools.by_id(int(p)).document(int(r), int(n)), int(p == AI)) for p, r, n in zip(self.pool[start:stop], self.row[start:stop], self.length[start:stop])]

    def summary(self, sequence_len: int) -> PlanSummary:
        rows_per_step = self.step_tokens // (sequence_len + 1)
        ai = self.ai
        _, first = np.unique((self.pool.astype(np.int64) << 48) | self.row, return_index=True)
        ai_tokens = int(self.length[ai].sum())
        return PlanSummary(
            name=self.spec.name,
            arm=self.spec.arm,
            seed=self.spec.seed,
            steps=self.steps,
            human_steps=self.spec.human_steps,
            step_tokens=self.step_tokens,
            trained_tokens=self.steps * rows_per_step * sequence_len,
            human_documents=int((~ai[first]).sum()),
            human_tokens=int(self.length[~ai].sum()),
            ai_documents=int(ai[first].sum()),
            ai_tokens=ai_tokens,
            ai_token_share=ai_tokens / int(self.length.sum()),
            passes=self.steps / self.spec.human_steps if self.spec.arm == Arm.REPEAT else 1.0,
        )


def _cut(lengths: np.ndarray, budget: int) -> np.ndarray:
    """Tokens used from each of the fewest leading documents that hold exactly `budget` tokens (the last one cut)."""
    cumulative = np.cumsum(lengths)
    count = int(np.searchsorted(cumulative, budget, side="left")) + 1
    if count > len(cumulative):
        raise ValueError(f"not enough data: {int(cumulative[-1]) if len(cumulative) else 0:,} tokens available, {budget:,} needed")
    used = lengths[:count].copy()
    used[-1] -= int(cumulative[count - 1]) - budget
    return used


@dataclass(frozen=True)
class _Docs:
    """Documents before ordering."""

    pool: np.ndarray
    row: np.ndarray
    length: np.ndarray

    @classmethod
    def prefix(cls, pool_id: int, lengths: np.ndarray, start: int, budget: int) -> _Docs:
        """Documents from `start` on in key order holding exactly `budget` tokens (the last one cut to fit)."""
        try:
            used = _cut(lengths[start:], budget)
        except ValueError as error:
            raise ValueError(f"pool {POOL_NAMES[pool_id]!r} (from document {start:,}): {error}") from None
        return cls(np.full(len(used), pool_id, dtype=np.uint8), np.arange(start, start + len(used), dtype=np.int64), used)

    def __add__(self, other: _Docs) -> _Docs:
        return _Docs(np.concatenate([self.pool, other.pool]), np.concatenate([self.row, other.row]), np.concatenate([self.length, other.length]))

    def keys(self, pools: Pools) -> np.ndarray:
        keys = np.empty(len(self.row), dtype=np.uint64)
        for pool_id in np.unique(self.pool):
            mask = self.pool == pool_id
            keys[mask] = pools.by_id(int(pool_id)).keys[self.row[mask]]
        return keys

    def ordered(self, pools: Pools, seed: int, epoch: int = 0) -> _Docs:
        order = np.argsort(order_keys(self.keys(pools), seed, epoch), kind="stable")
        return _Docs(self.pool[order], self.row[order], self.length[order])

    def truncated(self, budget: int) -> _Docs:
        """The first documents (in the current order) holding exactly `budget` tokens."""
        used = _cut(self.length, budget)
        return _Docs(self.pool[: len(used)], self.row[: len(used)], used)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def order_keys(keys: np.ndarray, seed: int, epoch: int = 0) -> np.ndarray:
    """Training-order hash of each document for a seed (and pass over the data, for repeat runs)."""
    salt = _splitmix64(_splitmix64(np.array([seed], dtype=np.uint64)) ^ np.array([epoch], dtype=np.uint64))
    return _splitmix64(keys ^ salt)


def plan_run(spec: RunSpec, pools: Pools, recipe: Recipe) -> DocumentPlan:
    row_len = recipe.sequence_len + 1
    step_tokens = recipe.rows_per_step(spec.depth) * row_len

    def usable(pool_id: int) -> np.ndarray:
        return np.minimum(pools.by_id(pool_id).lengths, row_len)

    human = _Docs.prefix(HUMAN, usable(HUMAN), 0, spec.human_steps * step_tokens)
    added = (spec.steps - spec.human_steps) * step_tokens
    if spec.arm in HUMAN_ONLY_ARMS:
        docs = human.ordered(pools, spec.seed)
    elif spec.arm in (Arm.AI, Arm.NATURAL):
        docs = (human + _Docs.prefix(AI, usable(AI), 0, added)).ordered(pools, spec.seed)
    elif spec.arm == Arm.HUMAN:
        docs = (human + _Docs.prefix(HUMAN, usable(HUMAN), len(human.row), added)).ordered(pools, spec.seed)
    else:
        docs = _repeated(human, pools, spec.seed, spec.steps * step_tokens)
    assert int(docs.length.sum()) == spec.steps * step_tokens
    return DocumentPlan(spec, step_tokens, docs.pool, docs.row, docs.length)


def _repeated(control: _Docs, pools: Pools, seed: int, budget: int) -> _Docs:
    per_pass = int(control.length.sum())
    passes = [control.ordered(pools, seed, epoch) for epoch in range(budget // per_pass)]
    if budget % per_pass:
        passes.append(control.ordered(pools, seed, len(passes)).truncated(budget % per_pass))
    docs = passes[0]
    for extra in passes[1:]:
        docs = docs + extra
    return docs
