# Usage

Changedet uses [fire](https://github.com/google/python-fire) for its CLI. Basic usage commands are given below.


1. For running change detection on two images, you can run the following command

```
changedet --algo imgdiff run sample1.tif sample2.tif
```

2. Get more information on algorithm used

```
changedet --algo imgdiff algo --help
```

3. Get more info on changedet pipeline

```
changedet --help
```

4. List available algorithms

```
changedet list_algos
```