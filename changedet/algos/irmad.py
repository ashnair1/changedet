from .base import MetaAlgo
from .catalog import AlgoCatalog


@AlgoCatalog.register("irmad")
class IRMAD(MetaAlgo):
    """IRMAD algorithm"""

    @classmethod
    def run(cls, im1, im2):
        print("Ello IRMAD here")

    @classmethod
    def help(cls):
        print(cls.__doc__)
