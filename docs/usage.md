# Usage

Changedet uses [fire](https://github.com/google/python-fire) for its CLI so it's not really all that much different from argparse.


1. For running change detection on two images, you can run the following command

```
python changedet/cli.py --algo imgdiff run sample1.tif sample2.tif
```

2. Get more information on algorithm used

```
python changedet/cli.py --algo imgdiff help
```

3. Get more info on changedet pipeline

```
python changedet/cli.py help
```