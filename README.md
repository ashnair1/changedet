# changedet

[![tests](https://github.com/ashnair1/changedet/actions/workflows/tests.yml/badge.svg)](https://github.com/ashnair1/changedet/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ashnair1/changedet/branch/master/graph/badge.svg?token=CKJCFQ7WLJ)](https://codecov.io/gh/ashnair1/changedet)
[![style](https://github.com/ashnair1/changedet/actions/workflows/style.yml/badge.svg)](https://github.com/ashnair1/changedet/actions/workflows/style.yml)
[![docs](https://github.com/ashnair1/changedet/actions/workflows/docs.yml/badge.svg)](https://github.com/ashnair1/changedet/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A Python toolkit for classical change detection algorithms in remote sensing and geospatial imagery.

## Features

changedet implements 4 classical change detection algorithms:

- **Image Differencing** (`imgdiff`) - Simple pixel-wise difference
- **Change Vector Analysis** (`cva`) - Magnitude and direction of change with Otsu thresholding
- **Iterated PCA** (`ipca`) - Principal Component Analysis with iterative reweighting
- **IR-MAD** (`irmad`) - Iteratively Reweighted Multivariate Alteration Detection

All algorithms work with multi-temporal satellite/aerial imagery in GeoTIFF format.

## Installation

### From source

```bash
git clone https://github.com/ashnair1/changedet
cd changedet
pip install .
```

### For development

```bash
git clone https://github.com/ashnair1/changedet
cd changedet
poetry install --with dev,docs
```

## Quick Start

### List available algorithms

```bash
changedet list
```

### Run change detection

```bash
# Basic usage
changedet --algo imgdiff run image1.tif image2.tif

# Change Vector Analysis with Euclidean distance
changedet --algo cva run image1.tif image2.tif --distance euclidean

# IR-MAD with 10 iterations
changedet --algo irmad run image1.tif image2.tif --niter 10
```

### Get help

```bash
# General help
changedet --help

# Algorithm-specific help
changedet --algo cva algo_obj --help
```

## Documentation

Full documentation is available at [https://ashnair1.github.io/changedet/](https://ashnair1.github.io/changedet/)

## License

MIT License - see [LICENSE](LICENSE) file for details.
