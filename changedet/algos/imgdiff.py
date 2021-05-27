from .catalog import AlgoCatalog  # , _AlgoCatalog  # MetaAlgo


@AlgoCatalog.register(name="imgdiff")
class ImageDiff:
    """
    Calculate absolute difference map

    Builds a change map by calculating the
     absolute difference
    between image 1 & image 2
    """

    @classmethod
    def run(cls, im1, im2):
        """
        Args:
            im1 ([type]): [description]
            im2 ([type]): [description]
        """
        print("Ello IMDIFF here")

    @classmethod
    def help(cls):
        """Print out docstring"""
        print(cls.__doc__)
