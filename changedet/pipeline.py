"""Main module."""

from changedet.algos import AlgoCatalog
from changedet.utils import init_logger


class ChangeDetPipeline:
    """
    Basic pipeline for running change detection algorithms.

    Args:
        algo (str): Change detection algorithm to be used

    Attributes:
        algo (str): Change detection algorithm to be used
        logger (logging.Logger): Logger object

    """

    def __init__(self, algo):
        self.algo = AlgoCatalog.get(algo)
        self.logger = init_logger("changedet")

    # Remove this function. Image loading and preprocessing should be done internally.
    def load(self, im1, im2):
        print("loading images")

    def run(self, im1, im2, **kwargs):
        if not self.algo:
            raise AssertionError("Algorithm not specified")
        self.load(im1, im2)
        kwargs.update({"logger": self.logger})
        self.algo.run(im1, im2, kwargs)

    @classmethod
    def list_algos(cls):
        print(AlgoCatalog.list())
