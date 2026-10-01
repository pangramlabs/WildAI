"""Build evaluation sets (see :mod:`wildai.data.evalsets`).

    python -m wildai.data.evalsets.cli c4 cosmopedia fw22 paloma --output-dir data/eval
    python -m wildai.data.evalsets.cli fw26 --output-dir data/eval --natural-pool data/pools/natural \
        --training-pools data/pools/human data/pools/ai
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wildai.data.config import default_config, load_config
from wildai.data.evalsets.common import EvalSetsConfig
from wildai.data.evalsets.fw22 import build_fw22
from wildai.data.evalsets.fw26 import build_fw26
from wildai.data.evalsets.hub import build_hub_set
from wildai.data.evalsets.paloma import build_paloma

SETS = ("c4", "cosmopedia", "fw22", "fw26", "paloma")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", choices=SETS)
    parser.add_argument("--config", type=Path, default=default_config("evalsets.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--natural-pool", type=Path, help="fw26: the natural pool directory")
    parser.add_argument("--training-pools", type=Path, nargs="*", default=[], help="fw26: pools FW26 must not overlap")
    args = parser.parse_args(argv)
    config = load_config(args.config, EvalSetsConfig)
    for name in args.sets:
        if name in ("c4", "cosmopedia"):
            manifests = [build_hub_set(name, getattr(config, name), args.output_dir)]
        elif name == "fw22":
            manifests = [build_fw22(config.fw22, args.output_dir)]
        elif name == "fw26":
            if args.natural_pool is None or not args.training_pools:
                parser.error("fw26 needs --natural-pool and --training-pools")
            manifests = build_fw26(config.fw26, args.natural_pool, args.training_pools, args.output_dir)
        else:
            manifests = build_paloma(config.paloma, args.output_dir)
        for manifest in manifests:
            print(f"{manifest.name}: {manifest.documents:,} documents, {manifest.gpt2_tokens:,} GPT-2 tokens")


if __name__ == "__main__":
    main()
