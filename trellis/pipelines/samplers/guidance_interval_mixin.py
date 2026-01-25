from __future__ import annotations

from typing import Any, Tuple

from .base import Sampler


class GuidanceIntervalSamplerMixin(Sampler):
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(
        self,
        model: Any,
        x_t: Any,
        t: float,
        cond: Any,
        neg_cond: Any,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        **kwargs: Any,
    ) -> Any:
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
