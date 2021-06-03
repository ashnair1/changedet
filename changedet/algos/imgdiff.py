from pathlib import Path

import rasterio as rio

from .base import MetaAlgo
from .catalog import AlgoCatalog


@AlgoCatalog.register("imgdiff")
class ImageDiff(MetaAlgo):
    """
    Calculate difference map

    Builds a change map by calculating the difference between image 1 & image 2
    """

    @classmethod
    def run(cls, im1, im2, flags):
        """Run Image Differencing algorithm

        Args:
            im1 (str): Path to image 1
            im2 (str): Path to image 2
            flags (dict): Flags for the algorithm

        Note: Image Differencing does not use flags
        """
        logger = flags.get("logger", None)
        if Path(im1).exists() & Path(im2).exists():
            im1 = rio.open(im1)
            im2 = rio.open(im2)
            arr1 = im1.read()
            arr2 = im2.read()

            # Calculate difference map
            logger.info("Calculating difference map")
            diff = arr1 - arr2

            outfile = "diffmap.tif"

            with rio.Env():
                profile = im1.profile
                with rio.open(outfile, "w", **profile) as dst:
                    dst.write(diff)
            logger.info("Change map written to %s", outfile)
