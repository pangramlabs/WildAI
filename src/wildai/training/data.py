"""Training batches: a run's documents, packed into rows and split across GPUs.

Every rank packs the full row stream (packing is cheap once documents are tokenized) and keeps its own share: step
`s` uses rows `[s * R, (s + 1) * R)` of the stream, rank `k` of `W` takes `R / W` consecutive rows of them and feeds
them in micro-batches of `device_batch_size` rows. The gradient of a step is the mean over its R rows whatever W is,
so the data order does not depend on the number of GPUs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from wildai.training.mixture import DocumentPlan, Pools
from wildai.training.packing import BestFitPacker, Piece, row_tokens


@dataclass(frozen=True)
class MicroBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


class TrainBatches:
    """Iterate steps; each step is this rank's list of micro-batches."""

    def __init__(
        self,
        plan: DocumentPlan,
        pools: Pools,
        sequence_len: int,
        rows_per_step: int,
        device_batch_size: int,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        rows_per_rank, rest = divmod(rows_per_step, world_size)
        if rest or rows_per_rank % device_batch_size:
            raise ValueError(f"{rows_per_step} rows per step do not split into {world_size} GPUs x micro-batches of {device_batch_size}")
        self.plan = plan
        self.pools = pools
        self.row_len = sequence_len + 1
        self.rows_per_step = rows_per_step
        self.rows_per_rank = rows_per_rank
        self.device_batch_size = device_batch_size
        self.rank = rank
        self.device = device

    @property
    def grad_accum_steps(self) -> int:
        return self.rows_per_rank // self.device_batch_size

    def __iter__(self) -> Iterator[list[MicroBatch]]:
        packer = iter(BestFitPacker(self.plan.pieces(self.pools), self.row_len, keep_remainder=True))
        for _ in range(self.plan.steps):
            rows = [next(packer) for _ in range(self.rows_per_step)]
            own = rows[self.rank * self.rows_per_rank : (self.rank + 1) * self.rows_per_rank]
            yield [self._micro_batch(own[i : i + self.device_batch_size]) for i in range(0, len(own), self.device_batch_size)]
        if next(packer, None) is not None:
            raise RuntimeError("documents left over after the last step; the plan and the step count disagree")

    def _micro_batch(self, rows: list[list[Piece]]) -> MicroBatch:
        tokens = torch.from_numpy(np.stack([row_tokens(row) for row in rows]).astype(np.int64))
        if self.device.type == "cuda":
            tokens = tokens.pin_memory().to(self.device, non_blocking=True)
        return MicroBatch(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
