"""Pangram AI-text labels through the public Pangram API (https://docs.pangram.com).

* :mod:`.models` -- typed request and response models of the documented REST API;
* :mod:`.client` -- HTTP client with retries and backoff (API key from ``PANGRAM_API_KEY``);
* :mod:`.batching` -- packing documents into Bulk API jobs under the billable-unit limit;
* :mod:`.cache` -- resumable per-shard cache of results and submitted jobs;
* :mod:`.labeler` -- labels a shard of documents, reusing cached results;
* :mod:`.cli` -- ``python -m wildai.labeling.pangram.cli`` over a document directory.
"""
