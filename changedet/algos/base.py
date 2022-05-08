from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np


class MetaAlgo(metaclass=ABCMeta):
    """Base class for an algorithm"""

    @abstractmethod
    @classmethod
    def run(cls, im1: np.ndarray, im2: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to run change detection

        Every algorithm will have at least one keyword argument - the global logger
        """
        assert "logger" in kwargs
