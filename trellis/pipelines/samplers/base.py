from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(
        self,
        model: Any,
        **kwargs: Any,
    ):
        """
        Sample from a model.
        """
        pass

    def _inference_model(self, model: Any, x_t: Any, t: float, cond: Any = None, **kwargs: Any) -> Any:
        return model(x_t, t, cond, **kwargs)
    