# Deterministic per-controller RNG.

from __future__ import annotations
import hashlib
import numpy as np


def make_rng(controller_id: str, global_seed: int) -> np.random.Generator:
    """Derive a deterministic Generator from (global_seed, controller_id).

    Same inputs always yield the same RNG stream. Changing one controller's
    id (or the global seed) changes only that controller's stream.
    """
    h = hashlib.blake2b(
        controller_id.encode("utf-8"),
        digest_size=8,
        key=global_seed.to_bytes(8, "little", signed=False),
    ).digest()
    seed = int.from_bytes(h, "little", signed=False)
    return np.random.default_rng(seed)
