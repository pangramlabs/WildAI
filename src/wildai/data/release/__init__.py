"""Hugging Face dataset release of WildAI: table schemas, export, dataset card and upload.

Configs (one directory each in the release folder):

* ``human``, ``ai``, ``mixed`` -- the pool documents with their labels (:class:`.schema.PoolRecord`);
* ``labels`` -- every pool document's metadata and labels without text;
* ``monthly_sample`` -- the monthly measurement sample (:class:`.schema.MonthlyRecord`);
* ``filter_audit`` -- the filter-survival audit on FineWeb's pipeline, split ``fineweb`` (:class:`.schema.FilterAuditRecord`).
  The DCLM side of the audit is released only as its per-stage survival counts (``results/web/filter_audit.json``).

The dataset, like the models, is released under CC BY-NC-SA 4.0.

Text that never went through FineWeb's PII step is anonymized on export (:mod:`wildai.data.pii`), after labeling; such
rows carry ``pii_anonymized_after_labeling = True``. Per-window Pangram scores are exported only with
``--include-pangram-windows``.

    python -m wildai.data.release.export pools --pools-dir data/pools --weborganizer-dir data/labels/weborganizer/pools \
        --release-dir release
    python -m wildai.data.release.card --release-dir release
    python -m wildai.data.release.push --release-dir release            # private repo under the pangram org
"""
