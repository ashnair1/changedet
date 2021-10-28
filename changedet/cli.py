"""Console script for changedet."""

import fire

from changedet.pipeline import ChangeDetPipeline


def main() -> None:
    # fire.Fire({"help": help, "test": test, "cdet": cdet})
    # cdet = ChangeDetPipeline()
    fire.Fire(ChangeDetPipeline)


if __name__ == "__main__":
    main()  # pragma: no cover
