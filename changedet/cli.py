"""Console script for changedet."""

import fire

from changedet.pipeline import ChangeDetPipeline


def help():
    print("changedet")
    print("=" * len("changedet"))
    print("Change Detection Toolbox")


def test():
    print("Hey there")


def cdet(algo, x, y):
    print(f"Running {algo} to find difference between {x} & {y}")


def main():
    # fire.Fire({"help": help, "test": test, "cdet": cdet})
    # cdet = ChangeDetPipeline()
    fire.Fire(ChangeDetPipeline)


if __name__ == "__main__":
    main()  # pragma: no cover
