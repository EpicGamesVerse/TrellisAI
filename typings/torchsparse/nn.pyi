from __future__ import annotations

from typing import Any, Sequence


class Conv3d:
    out_channels: int
    stride: Sequence[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        bias: bool = True,
        transposed: bool = False,
    ) -> None: ...

    def __call__(self, x: Any) -> Any: ...
