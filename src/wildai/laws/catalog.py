"""The benchmarked laws in the paper's table order, keyed by their published names."""

from __future__ import annotations

from wildai.laws.comparators import CD, Atlas, Chinchilla, Hamidieh, He, Jain, Lovelace, Muennighoff, Sedova, ShukorAdditive, ShukorJoint
from wildai.laws.forms import Law
from wildai.laws.ours import paper_law

PAPER_LAW = "ours_logshare_11"


def benchmark_laws() -> dict[str, Law]:
    """The twelve laws of the benchmark tables, ours last."""
    laws: list[Law] = [Chinchilla(), Muennighoff(), CD(), Lovelace(), Atlas(), He(), Hamidieh(), ShukorAdditive(), ShukorJoint(), Sedova(), Jain(), paper_law()]
    return {law.key: law for law in laws}


def law(key: str) -> Law:
    return benchmark_laws()[key]
