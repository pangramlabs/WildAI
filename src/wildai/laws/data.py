"""Typed access to the released runs (``results/models.csv``) and their losses (``results/losses.csv``).

Every law works in the same normalised coordinates, computed from a run's actual token counts:
``n = N / 1e8`` (N counts the input embedding), ``d = D_H / 1e9``, ``r = D_AI / D_H`` and ``t = D_H / (20 N)``.

A run is paired with the human-only control of its group: every AI or fresh-human addition trains on exactly its
control's human documents, so the change in log loss against that control isolates the effect of the addition.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

Arm = Literal["control", "ai", "human", "natural", "filtered", "repeat"]
Split = Literal["fit", "held_out", "filtering", "repetition"]

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

# The paper's evaluation targets in table order, with the labels the tables use.
TARGET_LABELS: dict[str, str] = {
    "c4": "C4",
    "fw22": "FW22",
    "fw26_human": "FW26-H",
    "paloma": "Paloma",
    "fw26": "FW26",
    "fw26_ai": "FW26-AI",
    "cosmopedia": "Cosmo",
}
PAPER_TARGETS: tuple[str, ...] = tuple(TARGET_LABELS)
HUMAN_TARGETS: tuple[str, ...] = ("c4", "fw22", "fw26_human", "paloma")

# Model sizes by depth, as the paper names them.
SIZE_LABELS: dict[int, str] = {4: "19.9M", 6: "35.8M", 9: "86.2M", 12: "135M", 16: "268M", 20: "477M", 26: "973M"}


@dataclass(frozen=True)
class Run:
    """One trained model, as listed in ``results/models.csv``."""

    name: str
    group: str
    arm: Arm
    size: str
    depth: int
    n_params: int
    seed: int
    added_ratio: float | None
    human_tokens: float
    ai_tokens: float
    total_tokens: float
    steps: int
    split: Split

    @property
    def ratio(self) -> float:
        """Actual AI-to-human token ratio r."""
        return self.ai_tokens / self.human_tokens

    @property
    def human_tpp(self) -> float:
        """Human tokens per parameter, TPP_h."""
        return self.human_tokens / self.n_params

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Run:
        return cls(
            name=row["name"],
            group=row["group"],
            arm=row["arm"],  # type: ignore[arg-type]
            size=row["size"],
            depth=int(row["depth"]),
            n_params=int(row["n_params"]),
            seed=int(row["seed"]),
            added_ratio=float(row["added_ratio"]) if row["added_ratio"] else None,
            human_tokens=float(row["human_tokens"]),
            ai_tokens=float(row["ai_tokens"]),
            total_tokens=float(row["total_tokens"]),
            steps=int(row["steps"]),
            split=row["split"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Coordinates:
    """Normalised law coordinates, one entry per run (arrays of equal length)."""

    n: np.ndarray
    d: np.ndarray
    r: np.ndarray
    t: np.ndarray

    @classmethod
    def from_tokens(cls, n_params: np.ndarray | float, human_tokens: np.ndarray | float, ai_tokens: np.ndarray | float) -> Coordinates:
        n_params, human, ai = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (n_params, human_tokens, ai_tokens))
        n_params, human, ai = np.broadcast_arrays(n_params, human, ai)
        if np.any(n_params <= 0) or np.any(human <= 0) or np.any(ai < 0):
            raise ValueError("token counts must be positive (AI tokens non-negative)")
        return cls(n=n_params / 1e8, d=human / 1e9, r=ai / human, t=human / n_params / 20.0)

    @classmethod
    def of_runs(cls, runs: Sequence[Run]) -> Coordinates:
        return cls.from_tokens(
            np.array([x.n_params for x in runs], float),
            np.array([x.human_tokens for x in runs], float),
            np.array([x.ai_tokens for x in runs], float),
        )

    def __len__(self) -> int:
        return len(self.n)


@dataclass(frozen=True)
class Observations:
    """Runs with their loss (bits per byte) on one target, each paired with its own human-only control.

    ``control[i]`` is the index of run i's control within this set; a control points to itself.
    """

    target: str
    runs: tuple[Run, ...]
    bpb: np.ndarray
    control: np.ndarray = field(repr=False)

    @cached_property
    def coords(self) -> Coordinates:
        return Coordinates.of_runs(self.runs)

    @cached_property
    def is_control(self) -> np.ndarray:
        return np.array([x.arm == "control" for x in self.runs])

    @cached_property
    def arms(self) -> np.ndarray:
        return np.array([x.arm for x in self.runs])

    @cached_property
    def depths(self) -> np.ndarray:
        return np.array([x.depth for x in self.runs])

    @cached_property
    def ratios(self) -> np.ndarray:
        return self.coords.r

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def select(self, keep: Callable[[Run], bool]) -> Observations:
        """The runs for which ``keep`` holds, plus the controls they need."""
        chosen = {x.name for x in self.runs if keep(x)}
        needed = chosen | {self.runs[self.control[i]].name for i, x in enumerate(self.runs) if x.name in chosen}
        return Observations.paired(self.target, [x for x in self.runs if x.name in needed], [b for x, b in zip(self.runs, self.bpb) if x.name in needed])

    @classmethod
    def paired(cls, target: str, runs: Sequence[Run], bpb: Sequence[float]) -> Observations:
        """Pair every run with the control of its group; every group present must include its control."""
        controls = {x.group: i for i, x in enumerate(runs) if x.arm == "control"}
        missing = sorted({x.group for x in runs if x.group not in controls})
        if missing:
            raise ValueError(f"groups without their control: {missing[:5]}")
        control = np.array([controls[x.group] for x in runs])
        return cls(target=target, runs=tuple(runs), bpb=np.asarray(bpb, float), control=control)


class Study:
    """All released runs and losses."""

    def __init__(self, runs: Iterable[Run], losses: dict[tuple[str, str], float]) -> None:
        self.runs: dict[str, Run] = {x.name: x for x in runs}
        self.losses = losses

    @classmethod
    def load(cls, results_dir: Path = RESULTS_DIR) -> Study:
        with (results_dir / "models.csv").open() as f:
            runs = [Run.from_row(row) for row in csv.DictReader(f)]
        with (results_dir / "losses.csv").open() as f:
            losses = {(row["name"], row["target"]): float(row["bpb"]) for row in csv.DictReader(f)}
        return cls(runs, losses)

    def split(self, *splits: Split) -> list[Run]:
        return [x for x in self.runs.values() if x.split in splits]

    def bpb(self, name: str, target: str) -> float:
        return self.losses[(name, target)]

    def covers(self, target: str, runs: Iterable[Run]) -> bool:
        """Whether every run has a loss on ``target`` (the filtering and repetition runs have only the first six targets)."""
        return all((x.name, target) in self.losses for x in runs)

    def observations(self, target: str, runs: Iterable[Run]) -> Observations:
        runs = list(runs)
        return Observations.paired(target, runs, [self.losses[(x.name, target)] for x in runs])

    def cohort(self, target: str) -> Observations:
        """The 800 runs of the scaling-law study (fitted and held-out sizes) on one target."""
        return self.observations(target, self.split("fit", "held_out"))
