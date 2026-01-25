from __future__ import annotations

import torch

from .pytorch.hilbert import decode as _hilbert_decode
from .pytorch.hilbert import encode as _hilbert_encode
from .pytorch.z_order import key2xyz as _key2xyz
from .pytorch.z_order import xyz2key as _xyz2key


def z_order_encode(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # 30-bit code => 10 bits per dimension
    return _xyz2key(x, y, z, b=None, depth=10)


def z_order_decode(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z, _ = _key2xyz(code, depth=10)
    return x, y, z


def hilbert_encode(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # 30-bit code => 10 bits per dimension
    locs = torch.stack([x, y, z], dim=-1)
    return _hilbert_encode(locs, num_dims=3, num_bits=10)


def hilbert_decode(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    locs = _hilbert_decode(code, num_dims=3, num_bits=10)
    return locs[:, 0], locs[:, 1], locs[:, 2]
