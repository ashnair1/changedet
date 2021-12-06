from typing import Any, Tuple

import numpy as np

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog


def calc_cvs(
    im1: np.ndarray, im2: np.ndarray, distance: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    diffmap = im2 - im1
    if distance == "manhattan":
        # Manhattan distance/L1 norm
        # mag = np.sum(np.abs(diffmap),axis=0)
        mag = np.linalg.norm(diffmap, ord=1, axis=0)
    else:
        # Euclidean distance/L2 norm
        # mag = np.sqrt(np.sum(np.square(diffmap), axis=0))
        mag = np.linalg.norm(diffmap, ord=2, axis=0)
    theta = np.arccos(diffmap / mag)
    return mag, theta


def otsu_thresh(im: np.ndarray, bins: int = 256) -> float:
    # Adapted from scikit image's otsu threshold

    counts, bin_edges = np.histogram(im, bins=bins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = float(bin_centers[idx])
    return threshold


def appy_threshold(dmap: np.ndarray) -> np.ndarray:
    bcm = np.zeros(dmap.shape)
    thresh = otsu_thresh(dmap)
    bcm[dmap > thresh] = 255
    return bcm


@AlgoCatalog.register("cva")
class CVA(MetaAlgo):
    """
    Calculate change vectors

    Builds a change map by calculating the amplitude map of the change vectors
    """

    @classmethod
    def run(cls, im1: np.ndarray, im2: np.ndarray, **flags: Any) -> np.ndarray:
        """Run Image Differencing algorithm

        Args:
            im1 (np.ndarray): Image 1 array
            im2 (np.ndarray): Image 2 array
            flags (dict): Flags for the algorithm
        """
        distance = flags.get("distance", "euclidean")
        logger = flags["logger"]

        assert distance in ["euclidean", "manhattan"]

        # Calculate change vectors
        logger.info("Calculating change vectors")
        mag, theta = calc_cvs(im1, im2, distance)
        bcm = appy_threshold(mag)

        return bcm
