<h1> Usage </h1>

Changedet uses [fire](https://github.com/google/python-fire) for its CLI. Basic usage commands are given below.


- For running change detection on two images, you can run the following command

```
changedet --algo imgdiff run sample1.tif sample2.tif
```

- Get more information on algorithm used

```
changedet --algo imgdiff algo_obj --help
```

- Get more info on changedet pipeline

```
changedet --help
```

- List available algorithms

```
changedet list
```