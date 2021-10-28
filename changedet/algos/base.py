from abc import ABCMeta, abstractclassmethod
from typing import Any

import numpy as np


class MetaAlgo(metaclass=ABCMeta):
    """Base class for an algorithm"""

    @abstractclassmethod
    def run(cls, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to run change detection

        Every algorithm will have at least one keyword argument - the global logger
        """
        assert kwargs is True
