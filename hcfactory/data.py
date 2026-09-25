"""Token-shard data pipeline (nanoGPT style: flat uint16 .bin files)."""
from __future__ import annotations

import os

import numpy as np
import torch


class TokenLoader:
    """Samples random contiguous windows from a memory-mapped token file.

    Under DDP each rank uses a different seed, so ranks see different windows.
    """

    def __init__(self, path: str, batch_size: int, seq_len: int, seed: int = 0, device="cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python -m hcfactory.prepare_data` first")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size, self.seq_len = batch_size, seq_len
        self.rng = np.random.default_rng(seed)
        self.device = device

    def __len__(self):
        return len(self.data)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        buf = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
        t = torch.from_numpy(buf)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        else:
            t = t.to(self.device)
        return t[:, :-1], t[:, 1:]

    def fixed_batches(self, n: int, seed: int = 1234):
        """Deterministic evaluation batches (identical for every variant)."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ix = rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
            buf = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
            t = torch.from_numpy(buf).to(self.device)
            out.append((t[:, :-1], t[:, 1:]))
        return out
