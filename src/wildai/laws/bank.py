"""Fitting many (law, target, fold) cases in parallel.

Every case's multi-start is split into chunks of starts that a process pool solves independently; the best solution of each
case is then selected exactly as :func:`wildai.laws.fit.fit` would, so results do not depend on the number of workers.

Folds name the fitted runs: ``all`` is the 726 runs at 19.9M to 268M (the 477M and 973M runs are held out), and
``without_<size>`` also leaves out one fitted size (leave-one-size-out).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from wildai.laws.catalog import benchmark_laws
from wildai.laws.data import RESULTS_DIR, SIZE_LABELS, Observations, Run, Study
from wildai.laws.fit import DEFAULT_OPTIONS, FitOptions, LawFit, Solution, chinchilla_backbone, select, solve, starts_for
from wildai.laws.forms import Law
from wildai.laws.ours import ABLATION_FORMS, ablation_law

FITTED_DEPTHS = (4, 6, 9, 12, 16)
FOLDS: tuple[str, ...] = ("all", *(f"without_{SIZE_LABELS[d]}" for d in FITTED_DEPTHS))
CHUNK = 8  # starts per pool task
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@cache
def resolve_law(key: str) -> Law:
    """A benchmark law by its published key, or a law-form ablation variant by its name."""
    laws = benchmark_laws()
    if key in laws:
        return laws[key]
    if key in ABLATION_FORMS:
        return ablation_law(key)
    raise KeyError(key)


def fold_depth(fold: str) -> int | None:
    """The size a fold leaves out (None for ``all``)."""
    if fold == "all":
        return None
    label = fold.removeprefix("without_")
    return next(d for d, s in SIZE_LABELS.items() if s == label)


def fold_runs(study: Study, fold: str) -> list[Run]:
    left_out = fold_depth(fold)
    return [x for x in study.split("fit") if x.depth != left_out]


@dataclass(frozen=True)
class Case:
    law: str
    target: str
    fold: str = "all"


# Per-process state of the pool workers (set by _init): the study, the options and cached training sets and starts.
_study: Study | None = None
_options = DEFAULT_OPTIONS


def _init(results_dir: Path, options: FitOptions) -> None:
    global _study, _options
    _study, _options = Study.load(results_dir), options
    _training.cache_clear()
    _starts.cache_clear()


@cache
def _training(target: str, fold: str) -> tuple[Observations, dict[str, float]]:
    assert _study is not None
    obs = _study.observations(target, fold_runs(_study, fold))
    return obs, chinchilla_backbone(obs, _options)


@cache
def _starts(case: Case) -> list[dict[str, float]]:
    return starts_for(resolve_law(case.law), _training(case.target, case.fold)[1], (), _options)


def _solve_chunk(task: tuple[Case, int, int]) -> tuple[Case, list[Solution]]:
    case, lo, hi = task
    obs, _backbone = _training(case.target, case.fold)
    return case, solve(resolve_law(case.law), obs, _starts(case)[lo:hi], _options, offset=lo)


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Workers solve small problems: one BLAS thread each avoids oversubscribing the machine (spawned workers read these)."""
    saved = {var: os.environ.get(var) for var in THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _select_case(item: tuple[Case, list[Solution]]) -> tuple[Case, LawFit]:
    case, solutions = item
    obs, _backbone = _training(case.target, case.fold)
    return case, select(resolve_law(case.law), obs, solutions, _options, case.fold)


def fit_cases(cases: Sequence[Case], results_dir: Path = RESULTS_DIR, workers: int = 1, options: FitOptions = DEFAULT_OPTIONS) -> dict[Case, LawFit]:
    """Fit every case; ``workers`` processes share the starts of all cases, then select and polish each case's best."""
    _init(results_dir, options)
    tasks = []
    for case in cases:
        count = len(_starts(case))
        tasks += [(case, lo, min(lo + CHUNK, count)) for lo in range(0, count, CHUNK)]
    tasks.sort(key=lambda task: -resolve_law(task[0].law).k)  # the slowest (largest) laws first, so no long task runs last
    solutions: dict[Case, list[Solution]] = defaultdict(list)
    if workers > 1:
        with _single_threaded_children(), mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(results_dir, options)) as pool:
            for case, found in pool.imap_unordered(_solve_chunk, tasks):
                solutions[case] += found
            fits = dict(pool.imap_unordered(_select_case, [(case, solutions[case]) for case in cases]))
    else:
        for task in tasks:
            case, found = _solve_chunk(task)
            solutions[case] += found
        fits = dict(map(_select_case, [(case, solutions[case]) for case in cases]))
    return {case: fits[case] for case in cases}
