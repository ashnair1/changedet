# Usage

changedet uses [Google Fire](https://github.com/google/python-fire) for its CLI.

## List Available Algorithms

```bash
changedet list
```

## Get Help

```bash
# General help
changedet --help

# Algorithm-specific help
changedet --algo imgdiff algo_obj --help
```

## Running Change Detection

### Basic Usage

```bash
changedet --algo ALGORITHM_NAME run image1.tif image2.tif
```

### Image Differencing

```bash
changedet --algo imgdiff run t1.tif t2.tif
```

### Change Vector Analysis

```bash
# Euclidean distance (default)
changedet --algo cva run t1.tif t2.tif

# Manhattan distance
changedet --algo cva run t1.tif t2.tif --distance manhattan
```

**Parameters:**
- `--distance`: `euclidean` or `manhattan`
- `--band`: Band index or -1 for all bands

### Iterated PCA

```bash
changedet --algo ipca run t1.tif t2.tif
```

**Parameters:**
- `--niter`: Number of iterations (default: 5)
- `--band`: Band index or -1 for all bands

### IR-MAD

```bash
changedet --algo irmad run t1.tif t2.tif
```

**Parameters:**
- `--niter`: Number of iterations (default: 10)
- `--sig`: Significance level (default: 0.0001)

## Output

All algorithms write output to `{algorithm}_cmap.tif` in the current directory.