from abc import ABCMeta, abstractclassmethod


class MetaAlgo(metaclass=ABCMeta):
    """ Base class for an algorithm """

    def __init__(self, **kwargs):
        """ Constructor """
        pass

    @abstractclassmethod
    def run(cls):
        """ Abstract method to run change detection """
        pass

    @abstractclassmethod
    def help(cls):
        """ Abstract method to return algorithm information """
        pass
