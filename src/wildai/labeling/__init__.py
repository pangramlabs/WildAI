"""Document labelers. Each writes a sidecar Parquet per input shard: ``id`` plus its label columns, row-aligned.

* :mod:`wildai.labeling.editlens` -- EditLens (Llama-3.2-3B) AI-editing buckets, used to preselect candidates;
* :mod:`wildai.labeling.pangram` -- Pangram document labels (Human / Mixed / AI) through the public Pangram API;
* :mod:`wildai.labeling.weborganizer` -- WebOrganizer topic and format labels.
"""
