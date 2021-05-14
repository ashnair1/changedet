"""Console script for changedet."""

import fire


def help():
    print("changedet")
    print("=" * len("changedet"))
    print("Change Detection Toolbox")

def test():
    print("Hey there")


def main():
    fire.Fire({"help": help,
               "test": test})


if __name__ == "__main__":
    main()  # pragma: no cover
