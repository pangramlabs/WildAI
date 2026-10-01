"""Control groups of the training grid, as the figures draw them.

A group is one human-only control plus the runs that add AI text or fresh human text on top of exactly its human
documents. Figures draw the groups of the matched-budget grid, whose controls have the same human tokens per parameter at
every size; groups at other budgets enter every fit but are not drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from paper.results import Model, controls, losses, models

MATCHED_BUDGETS = (4.72, 18.88, 37.76, 54.72, 72.97, 87.5)  # human tokens per parameter of the matched-budget grid
MATCHED_TOLERANCE = 0.015
FITTED_DEPTHS = (4, 6, 9, 12, 16)
HELD_OUT_DEPTHS = (20, 26)


def matched_budget(control: Model) -> float | None:
    """The matched-grid budget of a control, or None for a control at another budget."""

    return next((b for b in MATCHED_BUDGETS if abs(control.human_tpp - b) <= MATCHED_TOLERANCE * b), None)


@dataclass(frozen=True)
class Group:
    control: Model
    ai: tuple[Model, ...]  # by increasing AI ratio
    human: tuple[Model, ...]  # fresh-human additions, by increasing size

    @property
    def depth(self) -> int:
        return self.control.depth

    @property
    def budget(self) -> float | None:
        return matched_budget(self.control)


@cache
def groups() -> tuple[Group, ...]:
    members: dict[str, list[Model]] = {}
    for m in models().values():
        if m.arm in ("ai", "human"):
            members.setdefault(m.group, []).append(m)
    return tuple(Group(controls()[g], tuple(sorted((m for m in arms if m.arm == "ai"), key=lambda m: m.ratio)),
                       tuple(sorted((m for m in arms if m.arm == "human"), key=lambda m: m.human_tokens)))
                 for g, arms in members.items())


def matched_group(depth: int, budget: float) -> Group | None:
    """The matched-grid group at this size and budget with the most AI runs."""

    candidates = [g for g in groups() if g.depth == depth and g.budget == budget]
    return max(candidates, key=lambda g: len(g.ai), default=None)


def fresh_human_groups(depth: int, budget: float) -> list[Group]:
    """Every matched-grid group at this size and budget that has fresh-human additions."""

    return [g for g in groups() if g.depth == depth and g.budget == budget and g.human]


def added_fraction(model: Model, control: Model) -> float:
    """Tokens added per human token of the control: the AI ratio for AI runs, the extra human share for fresh-human runs."""

    return model.ratio if model.arm == "ai" else model.human_tokens / control.human_tokens - 1.0


def change_pct(model: Model, control: Model, target: str) -> float:
    """Percent change in loss on `target` against the control."""

    return 100.0 * (losses()[(model.name, target)] / losses()[(control.name, target)] - 1.0)
