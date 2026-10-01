"""Pack documents into Bulk API jobs that stay under the per-request billable-unit limit.

The Bulk API accepts at most 1,000 billable units per request; a unit is one started block of words per item (100 words
for Pangram 4, 1,000 for Pangram 3), with at least one unit per item. Packing with 100-word blocks is safe for both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from wildai.labeling.pangram.models import BulkItem


def billable_units(text: str, words_per_unit: int) -> int:
    return max(1, math.ceil(len(text.split()) / words_per_unit))


def pack_jobs(items: Sequence[BulkItem], *, max_units: int = 1000, words_per_unit: int = 100,
              max_items: int = 1000) -> list[list[BulkItem]]:
    """Consecutive groups of items, each within ``max_units`` billable units and ``max_items`` items."""

    jobs: list[list[BulkItem]] = []
    current: list[BulkItem] = []
    units = 0
    for item in items:
        cost = billable_units(item.text, words_per_unit)
        if cost > max_units:
            raise ValueError(f"document {item.id} needs {cost} units, more than a job allows ({max_units})")
        if current and (units + cost > max_units or len(current) == max_items):
            jobs.append(current)
            current, units = [], 0
        current.append(item)
        units += cost
    if current:
        jobs.append(current)
    return jobs
