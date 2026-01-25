from __future__ import annotations

from typing import Any

from .base import Sampler


class ClassifierFreeGuidanceSamplerMixin(Sampler):
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(
        self,
        model: Any,
        x_t: Any,
        t: float,
        cond: Any,
        neg_cond: Any,
        cfg_strength: float,
        **kwargs: Any,
    ) -> Any:
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
