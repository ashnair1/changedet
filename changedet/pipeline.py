from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.profiles import Profile

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

    def __init__(self, algo: str):
        """Initialise Pipeline

        Args:
            algo (str): Change detection algorithm to be used
        """
        self.algo_name = algo
        self.algo_obj = AlgoCatalog.get(algo)
        self.logger = init_logger("changedet")

    # Image loading and sanity checks should be done here
    def read(self, im1: str, im2: str, band: int) -> tuple[np.ndarray, np.ndarray]:
        """Read and prepare images

        Args:
            im1 (str): Path to image 1
            im2 (str): Path to image 2
            band (int): Band selection

        Raises:
            AssertionError: If images are not in the same projection system
            AssertionError: If images are not of same shape

        Returns:
            tuple:
                - arr1 (numpy.ndarray): Image 1 array of shape (B, H, W)
                - arr2 (numpy.ndarray): Image 2 array of shape (B, H, W)
        """
        try:
            assert Path(im1).exists() and Path(im2).exists()
        except AssertionError:
            self.logger.critical("Images not found")
            raise

        arr1, crs1, self.meta1 = self._read(im1, band)
        arr2, crs2, self.meta2 = self._read(im2, band)

        try:
            assert crs1 == crs2
        except AssertionError:
            self.logger.critical("Images are not in the same projection system")
            raise

        try:
            assert arr1.shape == arr2.shape
        except AssertionError:
            self.logger.critical("Image array shapes do not match")
            raise

        return arr1, arr2

    def _read(self, im: str, band: int) -> tuple[np.ndarray, CRS, Profile]:
        with rio.open(im) as raster:
            profile = raster.profile
            crs = raster.crs

            if band == -1:
                arr = raster.read()
            else:
                arr = np.expand_dims(raster.read(band), axis=0)
        return arr, crs, profile

    def run(self, im1: str, im2: str, band: int = -1, **kwargs: Any) -> None:
        """
        Run change detection on images

        Args:
            im1 (str): Path to image 1
            im2 (str): Path to image 2
            band (int): Band selection

        Raises:
            AssertionError: If no algorithm is specified
        """
        if not self.algo_obj:
            raise AssertionError("Algorithm not specified")
        im1a, im2a = self.read(im1, im2, band)
        # TODO: Decide whether algos should have their own loggers
        kwargs.update({"logger": self.logger, "band": band})
        cmap = self.algo_obj.run(im1a, im2a, **kwargs)
        self.write(cmap)

    def write(self, cmap: np.ndarray) -> None:
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
    def list(cls) -> None:
        """List available algorithms"""
        print(AlgoCatalog.list())
