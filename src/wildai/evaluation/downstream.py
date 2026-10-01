"""Downstream scores of a checkpoint for the paper's downstream figure: CORE (centered, mean over its tasks) and
5-shot MMLU accuracy.

    python -m wildai.evaluation.downstream --model 268m-g072-ai-r0.5 --bundle-dir eval_bundle/ --output downstream.json

`--bundle-dir` holds nanochat's public CORE eval bundle (downloaded there on first use); MMLU comes from the Hugging
Face dataset `cais/mmlu`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildai.evaluation.core import evaluate_core
from wildai.evaluation.mmlu import evaluate_mmlu
from wildai.evaluation.runtime import add_model_arguments, load_from_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="CORE eval bundle directory")
    parser.add_argument("--tasks", nargs="+", default=["core", "mmlu"], choices=["core", "mmlu"])
    parser.add_argument("--max-per-task", type=int, default=-1, help="CORE examples per task (-1: all, as in the paper)")
    parser.add_argument("--output", type=Path, help="write the scores as JSON")
    args = parser.parse_args()

    loaded = load_from_arguments(args)
    payload: dict[str, object] = {"model": args.model or str(args.checkpoint), "step": loaded.step}
    if "core" in args.tasks:
        core = evaluate_core(loaded, args.bundle_dir, args.max_per_task)
        payload["core"] = core.core
        payload["core_tasks"] = {k: v.model_dump() for k, v in core.tasks.items()}
        print(f"CORE: {core.core:.4f}")
    if "mmlu" in args.tasks:
        mmlu = evaluate_mmlu(loaded)
        payload["mmlu"] = mmlu.accuracy
        print(f"MMLU (5-shot): {mmlu.accuracy:.4f} ({mmlu.correct}/{mmlu.total})")
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
