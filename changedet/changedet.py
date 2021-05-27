"""Main module."""

from algos import AlgoCatalog


class ChangeDetPipeline:
    """
    Basic pipeline for running change detection algorithms.

    Args:
        algo (str): Change detection algorithm to be used

    """

    def __init__(self, algo=None):
        self.algo = AlgoCatalog.get(algo)

    def load(self, im1, im2):
        print("loading images")

    def run(self, im1, im2):
        if not self.algo:
            raise AssertionError("Algorithm not specified")
        self.load(im1, im2)
        self.algo.run(im1, im2)

    def list_algos(self):
        import pdb

        pdb.set_trace()

    def help(self):
        docstring = self.algo.__doc__ if self.algo else self.__doc__
        print(docstring)
