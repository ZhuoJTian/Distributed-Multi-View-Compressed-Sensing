# Distributed Multi-View Sparse Vector Recovery

Python implementation accompanying the paper **“Distributed Multi-View Sparse Vector Recovery”**, published in *IEEE Transactions on Signal Processing* (2023).

This maintained version reorganizes the original research code into a `src/` package while preserving the published numerical routines and the two original experiment workflows.

## Repository structure

```text
.
├── src/dmvcs/
│   ├── algorithms/       # ADMM and distributed reconstruction routines
│   ├── data/             # Data generation, processing, and validation
│   ├── inference/        # View/blockage inference
│   ├── network/          # Network generation
│   ├── optimization/     # CVXPY and numerical update routines
│   ├── plotting/         # Plotting utilities
│   └── utils/            # Shared helpers
├── scripts/experiments/
│   ├── compare_measurements.py
│   └── compare_blockages.py
├── tests/
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
```

## Requirements

- Python 3.10
- NumPy
- SciPy
- CVXPY
- NetworkX
- Matplotlib

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate dmv-svr
```

### pip with a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

This stage intentionally does not include `pyproject.toml`, so `pip install .` is not yet available. For interactive imports, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Python imports

```python
from dmvcs.algorithms import admm
from dmvcs.data.generation import generate_MVSVR_noise
from dmvcs.network.generation import gener_net_erdos
```

## Running experiments

Run commands from the repository root.

### Compare measurement counts

```bash
python scripts/experiments/compare_measurements.py
```


### Compare blockage dimensions

```bash
python scripts/experiments/compare_blockages.py
```


Both experiment modules now use a `main()` guard, so importing them does not start a long-running experiment.

## Required data directories

The original experiments still expect research data under these repository-relative paths:

```text
new_m_compareall_noise/Data_Sample/
new_d_compareall_noise/Data_Sample/
```

They save outputs under paths such as:

```text
new_m_compareall_noise/convergence2/
new_d_compareall_noise/glo/convergence2/
```

The input data are not included in this repository. A `FileNotFoundError` therefore usually means that the original generated data files have not been copied into the expected directory hierarchy.

## Plotting

The plotting implementation is available as a package module:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m dmvcs.plotting.plot_all_at_once
```

It expects the result files and directory layout used by the original research scripts.

## Validation

Compile all Python files without running experiments:

```bash
python -m compileall src scripts tests *.py
```

Run the smoke tests:

```bash
python -m unittest discover -s tests -v
```

The import smoke test is skipped when CVXPY is unavailable.

## Reproducibility notes

The original code uses both Python's `random` module and `numpy.random`. Set both seeds before generating data when deterministic behavior is required:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

Some experiments repeatedly solve convex optimization problems and can be computationally expensive. Solver behavior may differ across CVXPY versions and installed solver backends.


## Citation

```bibtex
@article{tian2023distributed,
  title={Distributed Multi-View Sparse Vector Recovery},
  author={Tian, Zhuojun and Zhang, Zhaoyang and Hanzo, Lajos},
  journal={IEEE Transactions on Signal Processing},
  volume={71},
  pages={1448--1463},
  year={2023},
  publisher={IEEE}
}
```

## License

This repository currently includes the MIT License. Before public release, the repository owner should confirm that all copyright holders agree to this license.
