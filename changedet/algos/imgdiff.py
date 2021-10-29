from typing import Any

import numpy as np

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog


@AlgoCatalog.register("imgdiff")
class ImageDiff(MetaAlgo):
    """
    Calculate difference map

    Builds a change map by calculating the difference between image 1 & image 2
    """

    @classmethod
    def run(cls, im1: np.ndarray, im2: np.ndarray, **flags: Any) -> np.ndarray:
        """Run Image Differencing algorithm

        Args:
            im1 (np.ndarray): Image 1 array
            im2 (np.ndarray): Image 2 array
            flags (dict): Flags for the algorithm
        """
        logger = flags.get("logger", None)

        # Calculate difference map
        logger.info("Calculating difference map")
        diff = im1 - im2

        return diff
