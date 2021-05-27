from collections import UserDict


class _AlgoCatalog(UserDict):
    """

    A global dictionary that stores information about the datasets
    and how to obtain them.
    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    """

    @staticmethod
    def register(name):
        import pdb

        pdb.set_trace()

        def _register(self, name, obj):
            self[name] = obj
            import pdb

            pdb.set_trace()

        return _register(name)  # , cls)

    # def register(self, name, obj):
    #     self[name] = obj

    def get(self, name):
        try:
            f = self[name]
        except KeyError as e:
            if isinstance(name, str):
                avail_algos = ", ".join(list(self.keys()))
                raise KeyError(
                    f"Algorithm {name} is not registered. \
                    Available algorithms are: {avail_algos}"
                ) from e
            else:
                f = None
        return f

    def list(self):
        """
        List all registered datasets.
        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)


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
