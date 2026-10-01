"""Held-out evaluation sets scored in bits per byte: C4, Cosmopedia, FW22, FW26 (with FW26-H and FW26-AI) and Paloma.

Every set is written as ``<output-dir>/<name>/shard_00000.parquet`` with a ``text`` column (Paloma adds ``domain``) and a
``manifest.json`` (:class:`EvalSetManifest`). Names match the targets in ``results/losses.csv``: ``c4``, ``cosmopedia``,
``fw22``, ``fw26``, ``fw26_human``, ``fw26_ai`` and ``paloma_<source>``. Sources are pinned to exact dataset revisions.

    python -m wildai.data.evalsets.cli c4 cosmopedia fw22 paloma --output-dir data/eval
    python -m wildai.data.evalsets.cli fw26 --output-dir data/eval --natural-pool data/pools/natural \
        --training-pools data/pools/human data/pools/ai
"""
