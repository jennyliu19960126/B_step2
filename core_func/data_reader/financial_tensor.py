"""Reader for the standalone financial-feature tensor artifact.

This module intentionally has no dependency on ``constant.params`` so it can
be used to inspect or align the GP input before the legacy data configuration
is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_financial_tensor(path: str | Path, feature_names: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Load ``X[quarter, feature, stock]`` and its axis metadata.

    ``feature_names`` optionally selects/reorders the feature axis by the
    Chinese names stored in the adjacent metadata file.  ``mmap_mode='r'``
    keeps the 818 MB source tensor out of RAM until it is actually consumed.
    """
    tensor_path = Path(path)
    metadata_path = tensor_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    X = np.load(tensor_path, mmap_mode="r")
    if tuple(metadata["shape"]) != X.shape:
        raise ValueError(f"metadata shape {metadata['shape']} does not match tensor shape {X.shape}")
    if feature_names is None:
        return X, metadata
    index = {name: i for i, name in enumerate(metadata["feature_names"])}
    missing = [name for name in feature_names if name not in index]
    if missing:
        raise KeyError(f"features are absent from tensor: {missing[:5]}")
    return X[:, [index[name] for name in feature_names], :], metadata
