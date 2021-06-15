from pathlib import Path

import numpy as np
import rasterio as rio

from changedet.algos import AlgoCatalog
from changedet.utils import init_logger


class ChangeDetPipeline:
    """
    Basic pipeline for running change detection algorithms.

    Args:
        algo (str): Change detection algorithm to be used

    Attributes:
        algo_name (str): Name of change detection algorithm
        algo_obj (str): Change detection algorithm object
        logger (logging.Logger): Logger object

    """

    def __init__(self, algo):
        """Initialise Pipeline

        Args:
            algo (str): Change detection algorithm to be used
        """
        self.algo_name = algo
        self.algo_obj = AlgoCatalog.get(algo)
        self.logger = init_logger("changedet")

    # Image loading and sanity checks should be done here
    def read(self, im1, im2):
        """Read and prepare images

        Args:
            im1 (str1): Path to image 1
            im2 (str1): Path to image 2

        Raises:
            AssertionError: If images are not in the same projection system
            AssertionError: Images are not of same shape

        Returns:
            tuple:
            - arr1 (numpy.ndarray): Image 1 array of shape (B, H, W)
            - arr2 (numpy.ndarray): Image 2 array of shape (B, H, W)
        """
        if Path(im1).exists() & Path(im2).exists():
            im1 = rio.open(im1)
            im2 = rio.open(im2)
            # Will be necessary for writing
            self.meta1 = im1.profile
            self.meta2 = im2.profile
            arr1 = im1.read()
            arr2 = im2.read()

            if im1.crs != im2.crs:
                self.logger.critical("Images are not in the same projection system.")
                raise AssertionError

            if im1.shape != im2.shape:
                self.logger.critical("Image array shapes do not match")
                raise AssertionError
            return arr1, arr2

    def run(self, im1, im2, **kwargs):
        """
        Run change detection on images

        Args:
            im1 (numpy.ndarray): Image 1 array of shape (B, H, W)
            im2 (numpy.ndarray): Image 2 array of shape (B, H, W)

        Raises:
            AssertionError: If no algorithm is specified
        """
        if not self.algo_obj:
            raise AssertionError("Algorithm not specified")
        im1a, im2a = self.read(im1, im2)
        # TODO: Decide whether algos should have their own loggers
        kwargs.update({"logger": self.logger})
        cmap = self.algo_obj.run(im1a, im2a, kwargs)
        self.write(cmap)

    def write(self, cmap):
        """Write change map to disk

        Args:
            cmap (numpy.ndarray): Change map of shape (B, H, W)

        """

        profile = self.meta1
        outfile = f"{self.algo_name}_cmap.tif"

        # Bandwise change or Single band change
        cmap = np.expand_dims(cmap, axis=0) if len(cmap.shape) == 2 else cmap

        profile["count"] = cmap.shape[0]

        with rio.Env():
            with rio.open(outfile, "w", **profile) as dst:
                for i in range(profile["count"]):
                    dst.write(cmap[i], i + 1)
        self.logger.info("Change map written to %s", outfile)

    @classmethod
    def list_algos(cls):
        """List available algorithms"""
        print(AlgoCatalog.list())
