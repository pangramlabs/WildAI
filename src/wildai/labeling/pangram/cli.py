"""Label a directory of document shards with Pangram through the public API.

    PANGRAM_API_KEY=... python -m wildai.labeling.pangram.cli --input-dir data/candidates --output-dir data/labels/pangram

Reads ``id`` and ``text`` of every ``*.parquet`` shard in ``--input-dir`` and writes a sidecar with the
:class:`wildai.labeling.pangram.models.PangramLabels` columns to ``--output-dir`` (same file names, same row order).
Per-shard caches under ``--cache-dir`` (default ``<output-dir>/.cache``) make the run resumable document by document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa

from wildai.data.arrow_schema import table_from_models
from wildai.data.config import default_config, load_config
from wildai.labeling.config import LabelingConfig
from wildai.labeling.pangram.cache import ShardCache
from wildai.labeling.pangram.client import PangramClient
from wildai.labeling.pangram.labeler import PangramLabeler
from wildai.labeling.pangram.models import PangramLabels
from wildai.labeling.sidecar import label_directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("labeling.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    settings = load_config(args.config, LabelingConfig).pangram
    client = PangramClient.from_env(settings.base_url)
    if settings.model not in client.list_models():
        parser.error(f"model selector {settings.model!r} is not available to this API key")
    labeler = PangramLabeler(client, settings)
    cache_root = args.cache_dir or args.output_dir / ".cache"

    def label(table: pa.Table, shard: Path) -> pa.Table:
        rows = labeler.label(table["id"].to_pylist(), table["text"].to_pylist(), ShardCache(cache_root / shard.stem))
        return table_from_models(rows, PangramLabels)

    label_directory(args.input_dir, args.output_dir, ["id", "text"], label,
                    shard_index=args.shard_index, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
