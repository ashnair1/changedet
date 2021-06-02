"""Main module."""

from changedet.algos import AlgoCatalog


class ChangeDetPipeline:
    """
    Basic pipeline for running change detection algorithms.

    Args:
        algo (str): Change detection algorithm to be used

    """

    def __init__(self, algo):
        self.algo = AlgoCatalog.get(algo)

    # Remove this function. Image loading and preprocessing should be done internally.
    def load(self, im1, im2):
        print("loading images")

    def run(self, im1, im2, **kwargs):
        if not self.algo:
            raise AssertionError("Algorithm not specified")
        self.load(im1, im2)
        self.algo.run(im1, im2, kwargs)

    def list_algos(self):
        print(AlgoCatalog.list())
