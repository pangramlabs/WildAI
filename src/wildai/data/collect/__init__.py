"""Collect English web documents into :class:`wildai.data.schema.WebDocument` Parquet.

Two sources, one output layout (``<output>/<dump>/part-*.parquet``):

* :mod:`wildai.data.collect.fineweb` samples documents from Hugging Face FineWeb dumps (crawls up to CC-MAIN-2025-26);
* :mod:`wildai.data.collect.common_crawl` runs the FineWeb recipe with DataTrove on Common Crawl WARC files for crawls
  FineWeb has not released.
"""
