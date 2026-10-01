"""The released models with what their cards need: the run, its reference runs and its held-out losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from wildai.laws.data import RESULTS_DIR, Run, Study


@dataclass(frozen=True)
class CatalogEntry:
    run: Run
    control: Run | None  # the group's human-only control, for runs that add to it
    partner: Run | None  # the other model of a filtering pair
    losses: dict[str, float]  # target -> bits per byte

    @property
    def ai_share(self) -> float:
        return self.run.ai_tokens / (self.run.human_tokens + self.run.ai_tokens)

    def release_metadata(self) -> dict[str, str | int | float | None]:
        """The ``release`` block of config.json."""
        run = self.run
        tokens = {"human_tokens": int(run.human_tokens), "ai_tokens": int(run.ai_tokens), "total_tokens": int(run.total_tokens)}
        return {**asdict(run), **tokens, "ai_ratio": run.ratio}


class Catalog:
    def __init__(self, study: Study) -> None:
        self.study = study
        self._losses: dict[str, dict[str, float]] = {}
        for (model, target), bpb in study.losses.items():
            self._losses.setdefault(model, {})[target] = bpb

    @classmethod
    def load(cls, results_dir: Path = RESULTS_DIR) -> Catalog:
        return cls(Study.load(results_dir))

    def names(self) -> list[str]:
        return list(self.study.runs)

    def entry(self, name: str) -> CatalogEntry:
        if name not in self.study.runs:
            raise KeyError(f"{name} is not in models.csv")
        run = self.study.runs[name]
        group = [x for x in self.study.runs.values() if x.group == run.group and x.name != name]
        control = next((x for x in group if x.arm == "control"), None) if run.arm in ("ai", "human", "repeat") else None
        partner = next((x for x in group if x.arm in ("natural", "filtered")), None) if run.arm in ("natural", "filtered") else None
        return CatalogEntry(run=run, control=control, partner=partner, losses=self._losses.get(name, {}))

    def entries(self) -> list[CatalogEntry]:
        return [self.entry(name) for name in self.study.runs]
