"""Minimal memory-mapped loader for the processed Haxby arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_haxby(root: str | Path | None = None, mmap_mode: str = "r") -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent / "processed"
    with (root / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {
        "train_X": np.load(root / "train_X.npy", mmap_mode=mmap_mode),
        "train_y": np.load(root / "train_y.npy"),
        "test_X": np.load(root / "test_X.npy", mmap_mode=mmap_mode),
        "test_y": np.load(root / "test_y.npy"),
        "train_metadata": pd.read_csv(root / "train_metadata.csv"),
        "test_metadata": pd.read_csv(root / "test_metadata.csv"),
        "mask": np.asarray(
            __import__("nibabel").load(root / "common_brain_mask_3mm.nii.gz").dataobj,
            dtype=bool,
        ),
        "manifest": manifest,
    }


def iter_run_clients(data: dict, split: str = "train"):
    """Yield ``(client_id, X, y, metadata)`` for each contiguous run/client."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    x = data[f"{split}_X"]
    y = data[f"{split}_y"]
    metadata = data[f"{split}_metadata"]
    for client_id, rows in metadata.groupby("client_id", sort=False):
        positions = rows.index.to_numpy()
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError(f"Client {client_id} is not stored contiguously")
        client_x = x[positions[0] : positions[-1] + 1]
        yield client_id, client_x, y[positions], rows.reset_index(drop=True)


if __name__ == "__main__":
    data = load_haxby()
    for name in ("train_X", "train_y", "test_X", "test_y", "mask"):
        print(name, data[name].shape, data[name].dtype)
