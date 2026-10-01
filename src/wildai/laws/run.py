"""Regenerate every law output of the paper in results/laws/.

    python -m wildai.laws.run [--targets c4 fw22 ...] [--workers N] [--out results/laws] [--reuse-fits]

Fits every benchmark law on every target, the leave-one-size-out fits and the law-form ablation on C4, then writes the
scores, predictions and derived quantities (see the package docstring for the file schemas). ``--reuse-fits`` skips the
fitting and recomputes everything else from an existing ``fits.json``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from wildai.laws.analysis import ablation, benchmark, ceg_ladder, filtering, masking, placement, recommendations, significance, stability
from wildai.laws.bank import Case, fit_cases, resolve_law
from wildai.laws.catalog import PAPER_LAW, benchmark_laws
from wildai.laws.data import HUMAN_TARGETS, PAPER_TARGETS, RESULTS_DIR, Study
from wildai.laws.fit import LawFit
from wildai.laws.records import FitRecord, read_fits, write_csv, write_json


def all_cases(targets: list[str]) -> list[Case]:
    cases = benchmark.cases(targets) + placement.cases(targets)
    if "c4" in targets:
        cases += stability.cases() + ablation.cases()
    return list(dict.fromkeys(cases))  # unique, in order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=list(PAPER_TARGETS), help="evaluation targets (default: the seven paper targets)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="directory with models.csv and losses.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "laws")
    parser.add_argument("--reuse-fits", action="store_true", help="read fits.json instead of fitting")
    args = parser.parse_args()
    study = Study.load(args.results)
    targets: list[str] = args.targets
    out: Path = args.out
    start = time.monotonic()

    cases = all_cases(targets)
    if args.reuse_fits:
        stored = {Case(r.law, r.target, r.fold): r.fit() for r in read_fits(out / "fits.json")}
        fits: dict[Case, LawFit] = {case: stored[case] for case in cases}
    else:
        fits = fit_cases(cases, args.results, args.workers)
        write_json(out / "fits.json", [FitRecord.of(resolve_law(c.law), f) for c, f in fits.items()])
    print(f"{len(fits)} fits ({time.monotonic() - start:.0f} s)", flush=True)

    scores, predictions = benchmark.score_fits(study, {c: f for c, f in fits.items() if c in set(benchmark.cases(targets))})
    if "c4" in targets:
        stable, left_out = stability.analyse(study, {c: fits[c] for c in stability.cases()})
        write_csv(out / "stability.csv", stable)
        predictions += left_out
        write_csv(out / "ablation.csv", ablation.analyse(study, {c: fits[c] for c in ablation.cases()}))
    write_csv(out / "scores.csv", scores)
    write_csv(out / "predictions.csv", predictions)

    human = [t for t in HUMAN_TARGETS if t in targets]
    if human:
        write_json(out / "significance.json", significance.analyse(predictions, list(benchmark_laws()), human))
    write_json(out / "harm_placement.json", placement.analyse(study, fits, targets))
    records, summary = filtering.analyse(study, fits, ["chinchilla_5", PAPER_LAW], [t for t in targets if study.covers(t, study.split("filtering"))])
    write_csv(out / "filtering.csv", records)
    write_csv(out / "filtering_summary.csv", summary)
    # No Paloma target: under our law's Paloma fits the loss of the 2026 mix stops falling with more tokens at a fixed size,
    # so the mix has no compute-equivalent gain there.
    records, curve, summary = ceg_ladder.analyse(study, fits, [t for t in targets if not t.startswith("paloma")])
    write_csv(out / "ceg_ladder.csv", records)
    write_csv(out / "ceg_ladder_curve.csv", curve)
    write_json(out / "ceg_ladder_summary.json", summary)
    costs, shares, values = recommendations.analyse(study, fits, targets)
    write_csv(out / "cost_of_not_filtering.csv", costs)
    write_csv(out / "optimal_share.csv", shares)
    write_csv(out / "token_value.csv", values)
    contrasts = masking.contrasts(study)
    write_csv(out / "masking_contrasts.csv", contrasts)
    write_json(out / "masking.json", masking.summarise(study, contrasts))
    print(f"wrote {out} ({time.monotonic() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
