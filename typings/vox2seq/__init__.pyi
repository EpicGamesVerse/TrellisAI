from __future__ import annotations

from typing import Sequence

import torch


def encode(
    coords: torch.Tensor,
    mode: str,
    permute: Sequence[int] | None = None,
) -> torch.Tensor: ...
