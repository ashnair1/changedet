from collections import UserDict


class _AlgoCatalog(UserDict):
    """

    A global dictionary that stores information about the algorithms used and
    their corresponding pipeline. It contains a mapping of algorithm names to
    the algorithm class object.

    >>> from changedet.algos import AlgoCatalog
    >>> import pprint
    >>> pprint.pprint(AlgoCatalog)
    {'imgdiff': <class 'changedet.algos.imgdiff.ImageDiff'>,
    'irmad': <class 'changedet.algos.irmad.IRMAD'>}


    """

    def register(self, name: str):
        def inner_wrapper(wrapped_class):
            if name in self.keys():
                raise AssertionError(f"Algorithm {name} already exists.")
            self[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get(self, name):
        try:
            f = self[name]
        except KeyError as e:
            if isinstance(name, str):
                avail_algos = ", ".join(list(self.keys()))
                raise KeyError(
                    f"Algorithm {name} is not registered. Available algorithms are: {avail_algos}"
                ) from e
            else:
                f = None
        return f

    def list(self):
        """
        List all registered algorithms.
        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)


# Instantiate AlgoCatalog
AlgoCatalog = _AlgoCatalog()


# class MetaAlgo(type):
#     def __new__(meta, name):
#         AlgoCatalog.register(name, cls)
#         return cls
#     # subclasses = []
#     # def __init_subclass__(cls, name, **kwargs):
#     #     super().__init_subclass__(**kwargs)
#     #     cls.subclasses.append(cls)
#     #     # import pdb; pdb.set_trace()
#     #     # AlgoCatalog.register(name, cls)

# def register(cls):
#     AlgoCatalog.register(name, cls)
#     return cls

# from abc import ABCMeta, abstractmethod


# class MetaAlgo(metaclass=ABCMeta):
#     """ Base class for an executor """

#     def __init__(self, **kwargs):
#         """ Constructor """
#         pass

#     @abstractmethod
#     def run(self, command: str) -> (str, str):
#         """ Abstract method to run a command """
#         pass


# class AlgoFactory:
#     """ The factory class for creating change detection algo"""

#     registry = {}
#     AlgoCatalog = _AlgoCatalog()
#     """ Internal registry for available executors """

#     @classmethod
#     def register(cls, name: str):
#         import pdb

#         pdb.set_trace()

#         def inner_wrapper(wrapped_class: MetaAlgo):
#             if name in cls.registry:
#                 print("HEY there")
#                 # logger.warning('Executor %s already exists. Will replace it', name)
#             cls.registry[name] = wrapped_class
#             return wrapped_class

#         return inner_wrapper

#     # end register()
