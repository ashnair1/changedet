from pathlib import Path

import numpy as np
import rasterio as rio

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog


def calc_cvs(im1, im2):
    diffmap = im2 - im1
    # Euclidean norm
    # mag = np.sqrt(np.sum(np.square(diffmap), axis=0))
    mag = np.linalg.norm(diffmap, axis=0)
    theta = np.arccos(diffmap / mag)
    return mag, theta


def otsu_thresh(im, bins=256):
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
    threshold = bin_centers[idx]

    return threshold


def appy_threshold(dmap):
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
    def run(cls, im1, im2, flags):
        """Run Image Differencing algorithm

        Args:
            im1 (str): Path to image 1
            im2 (str): Path to image 2
            flags (dict): Flags for the algorithm
        """
        logger = flags.get("logger", None)
        if Path(im1).exists() & Path(im2).exists():
            im1 = rio.open(im1)
            im2 = rio.open(im2)
            arr1 = im1.read()
            arr2 = im2.read()

            # Calculate change vectors
            logger.info("Calculating change vectors")
            mag, theta = calc_cvs(arr1, arr2)
            bcm = appy_threshold(mag)

            outfile = "cva_changemap.tif"

            with rio.Env():
                profile = im1.profile
                profile["count"] = 1
                with rio.open(outfile, "w", **profile) as dst:
                    dst.write(bcm, 1)
            logger.info("Change map written to %s", outfile)
