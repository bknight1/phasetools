# pyMAGEMin

Python wrappers around Julia `MAGEMin_C` for phase equilibria and garnet growth workflows.

## Installation

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Julia Setup

After Python package installation, run the explicit Julia environment setup/check helper:

```bash
pyMAGEMin-julia-setup --check
```

If `MAGEMin_C` is not installed yet, install it with:

```bash
pyMAGEMin-julia-setup --install
```

You can also run the setup helper module directly in Python code from:

- [src/pyMAGEMin/julia_setup.py](src/pyMAGEMin/julia_setup.py)

If Julia itself is missing, install it from:

- https://julialang.org/downloads/

## Quick verification

You can verify your Julia package setup with:

```bash
julia --version
julia -e 'using MAGEMin_C'
```
