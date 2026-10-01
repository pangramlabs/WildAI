"""Common Crawl access: a crawl's WARC file list, uniform WARC-file frames, and resumable downloads.

The sampling unit for everything we draw from raw Common Crawl is the WARC file. A frame is a set of WARC files drawn
uniformly without replacement from a crawl's complete ``warc.paths.gz`` list, so it covers the whole crawl period rather
than a prefix of the file list.
"""

from __future__ import annotations

import gzip
import os
import random
import shutil
import time
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wildai.data.hashing import keyed_hash, seeded_key

COMMON_CRAWL_BASE = "https://data.commoncrawl.org"
USER_AGENT = "wildai-data-pipeline"


class WarcFrame(BaseModel):
    """The WARC files of one crawl a run reads."""

    model_config = ConfigDict(frozen=True)

    dump: str
    seed: int
    crawl_files: int
    """Number of WARC files in the whole crawl."""
    paths: list[str]
    """Selected WARC paths (``crawl-data/...warc.gz``), in crawl-list order."""

    @property
    def inclusion_probability(self) -> float:
        return len(self.paths) / self.crawl_files


def _fetch(url: str, timeout: float = 300.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def crawl_warc_paths(dump: str) -> list[str]:
    """Every WARC file path of a crawl, from its published ``warc.paths.gz``."""

    raw = gzip.decompress(_fetch(f"{COMMON_CRAWL_BASE}/crawl-data/{dump}/warc.paths.gz")).decode("utf-8")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def select_frame(dump: str, paths: list[str], files: int | None, seed: int) -> WarcFrame:
    """``files`` WARC files drawn uniformly without replacement (``None`` keeps every file), in crawl-list order."""

    if files is None:
        return WarcFrame(dump=dump, seed=seed, crawl_files=len(paths), paths=list(paths))
    if not 0 < files <= len(paths):
        raise ValueError(f"cannot draw {files} WARC files from a crawl of {len(paths)}")
    rng = random.Random(keyed_hash(dump, seeded_key("warc-frame", seed)))
    chosen = sorted(rng.sample(range(len(paths)), files))
    return WarcFrame(dump=dump, seed=seed, crawl_files=len(paths), paths=[paths[i] for i in chosen])


def download(path: str, destination: Path, retries: int = 5) -> Path:
    """Download one Common Crawl file to ``destination`` (skipped if already complete), retrying with backoff."""

    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(f"{COMMON_CRAWL_BASE}/{path}", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=900) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=8 << 20)
            os.replace(partial, destination)
            return destination
        except OSError:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise AssertionError("unreachable")
