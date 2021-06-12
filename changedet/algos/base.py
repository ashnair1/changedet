from abc import ABCMeta, abstractclassmethod


class MetaAlgo(metaclass=ABCMeta):
    """Base class for an algorithm"""

    @abstractclassmethod
    def run(cls, **kwargs):
        """
        Abstract method to run change detection

        Every algorithm will have at least one keyword argument - the global logger
        """
        assert kwargs is True
        pass
