"""Train one WildAI model: a model from `results/models.csv` by name, or a new one from a budget.

The human and AI pools are tokenized first with `python -m wildai.training.pools` into `<data-dir>/human` and
`<data-dir>/ai`. Examples:

    # a paper model, on 8 GPUs
    torchrun --standalone --nproc_per_node=8 -m wildai.training.train --model 268m-g072-ai-r0.5 --data-dir data --out-dir checkpoints
    # a new model: depth 12, 20 human tokens per parameter, plus half as many AI tokens
    torchrun --standalone --nproc_per_node=8 -m wildai.training.train --depth 12 --tpp 20 --arm ai --ratio 0.5 --data-dir data --out-dir checkpoints
    # a filtering pair: 20 tokens per parameter of a web mix with the 2026 AI share, and the same mix without its AI text
    torchrun --standalone --nproc_per_node=8 -m wildai.training.train --depth 12 --tpp 20 --arm natural --data-dir data
    torchrun --standalone --nproc_per_node=8 -m wildai.training.train --depth 12 --tpp 20 --arm filtered --data-dir data
    # a budget from a spec file (see configs/training/), and a dry run that only prints the data plan
    python -m wildai.training.train --spec configs/training/smoke_d4.json --data-dir data --dry-run

A spec file holds the budget arguments (`depth`, `tpp`, `arm`, `ratio`, `ai_share`, `seed`, optional `name`);
`--recipe` takes a JSON file of `Recipe` field overrides (the defaults are the paper's recipe).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wildai.training.mixture import WEB_AI_SHARE_2026, Arm, Pools, RunSpec, plan_run
from wildai.training.recipe import Recipe
from wildai.training.registry import DEFAULT_MODELS_CSV, load_models, run_spec
from wildai.training.trainer import train


class BudgetSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int
    tpp: float
    arm: Arm
    ratio: float = 0.0
    ai_share: float = WEB_AI_SHARE_2026
    seed: int = 1337
    name: str | None = None

    def run_spec(self, recipe: Recipe) -> RunSpec:
        return RunSpec.from_budget(self.depth, self.arm, self.tpp, self.seed, recipe, self.ratio, self.ai_share, self.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", help="public model name from results/models.csv")
    which.add_argument("--spec", type=Path, help="JSON budget spec (see configs/training/)")
    which.add_argument("--depth", type=int, help="new model: transformer depth (4, 6, 9, 12, 16, 20, 26 in the paper)")
    parser.add_argument("--tpp", type=float, help="new model: human tokens per parameter (natural/filtered: web-mix tokens)")
    parser.add_argument("--arm", type=Arm, choices=[a.value for a in Arm], default=Arm.CONTROL, help="new model: arm")
    parser.add_argument("--ratio", type=float, default=0.0, help="new model: added tokens per human token (ai, human, repeat)")
    parser.add_argument("--ai-share", type=float, default=WEB_AI_SHARE_2026, help="new model: AI share of a web mix's tokens (natural, filtered)")
    parser.add_argument("--seed", type=int, default=1337, help="new model: seed (weights and data order)")
    parser.add_argument("--name", help="new model: run name (default derived from the budget)")
    parser.add_argument("--data-dir", type=Path, required=True, help="directory with the tokenized pools human/ and ai/")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"), help="checkpoints go to <out-dir>/<name>/")
    parser.add_argument("--models-csv", type=Path, default=DEFAULT_MODELS_CSV)
    parser.add_argument("--recipe", type=Path, help="JSON file of Recipe overrides")
    parser.add_argument("--device-batch-size", type=int, help="sequences per GPU per micro-step (memory only)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="print the data plan and exit")
    args = parser.parse_args()
    if args.depth is not None and args.tpp is None:
        parser.error("--depth needs --tpp")
    return args


def main() -> None:
    args = parse_args()
    overrides = json.loads(args.recipe.read_text(encoding="utf-8")) if args.recipe else {}
    if args.device_batch_size:
        overrides["device_batch_size"] = args.device_batch_size
    recipe = Recipe(**overrides)
    if args.model:
        spec = run_spec(args.model, load_models(args.models_csv))
    elif args.spec:
        spec = BudgetSpec.model_validate_json(args.spec.read_text(encoding="utf-8")).run_spec(recipe)
    else:
        budget = BudgetSpec(depth=args.depth, tpp=args.tpp, arm=args.arm, ratio=args.ratio, ai_share=args.ai_share, seed=args.seed, name=args.name)
        spec = budget.run_spec(recipe)
    pools = Pools.from_directory(args.data_dir)
    if args.dry_run:
        print(spec.model_dump_json(indent=2))
        print(plan_run(spec, pools, recipe).summary(recipe.sequence_len).model_dump_json(indent=2))
        return
    train(spec, recipe, pools, args.out_dir, args.log_every)


if __name__ == "__main__":
    main()
