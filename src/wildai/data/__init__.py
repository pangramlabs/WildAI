"""Build a WildAI-style corpus and the paper's web measurements.

Pipeline (every step is ``python -m wildai.<module> --help``; settings live in ``configs/data/``):

1. Collect web documents (:mod:`wildai.data.collect`): Hugging Face FineWeb dumps, and FineWeb's recipe run with DataTrove
   on Common Crawl WARC files for later crawls.
2. Label with EditLens (:mod:`wildai.labeling.editlens`) and preselect candidates (:mod:`wildai.data.preselect`).
3. Label candidates with Pangram through the public API (:mod:`wildai.labeling.pangram`).
4. Split them into hash-ordered training pools (:mod:`wildai.data.pools`; schema :class:`wildai.data.schema.PoolDocument`).
5. Build evaluation sets (:mod:`wildai.data.evalsets`).
6. Measure the web (:mod:`wildai.data.monthly_sample`, :mod:`wildai.data.measure`), with WebOrganizer topics and formats
   (:mod:`wildai.labeling.weborganizer`).
7. Export the Hugging Face release (:mod:`wildai.data.release`).

Collection with DataTrove (step 1's Common Crawl part, the monthly sample's later months and the filter audit) needs a
separate Python 3.10 environment pinned to FineWeb's tools; see :mod:`wildai.data.collect.recipe`.
"""
