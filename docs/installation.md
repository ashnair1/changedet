# Installation

## Requirements

- Python >=3.11, <3.13

## From Source

Clone the repository:

```bash
git clone https://github.com/ashnair1/changedet.git
cd changedet
```

Install with pip:

```bash
pip install .
```

## For Development

Using Poetry:

```bash
poetry install --with dev,docs
poetry shell
```

Run tests:

```bash
pytest
```

Run pre-commit checks:

```bash
pre-commit run --all-files
```

