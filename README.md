# Network DQ Ridesharing Simulation

This repository reproduces the ridesharing experiment described in the paper:

**Peng, Tianyi, Naimeng Ye, and Andrew Zheng. "Differences-in-Neighbors for Network Interference in Experiments." arXiv preprint arXiv:2503.02271 (2025).**

The project simulates and evaluates ridesharing dispatch and pricing policies using network-based difference-in-quantities (DQ) estimators, with a focus on causal inference and policy evaluation using switchback experiments.

## Features

- **Ridesharing simulation** using JAX and custom environments.
- **Switchback experiments** for robust policy evaluation.
- **Multiple estimators** (Naive, DQ) for treatment effect estimation.
- **Automated experiment workflow** using Snakemake.
- **Data analysis and plotting** in Python.

## Project Structure

- `rideshare.py`, `rideshare-incremental.py`: Main simulation scripts for running ridesharing experiments.
- `compute-ate.py`: Computes the Average Treatment Effect (ATE) from simulation results.
- `Snakefile`: Snakemake workflow for running all experiments and analyses.
- `plot.py`: Script for analyzing and plotting results.
- `output/`: Directory for simulation outputs and intermediate results.
- `pyproject.toml`, `poetry.lock`: Python dependencies (managed with Poetry).
- `manhattan-nodes.parquet`, `taxi-zones.parquet`: Data files for the simulation environment.

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- NVIDIA A100 GPU (required for optimal performance)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd dn-ridesharing
   ```

2. **Install Python dependencies:**
   ```bash
   poetry install
   ```

3. **Install JAX with CUDA support:**
   ```bash
   poetry run pip install jax[cuda]
   ```
   Note: The version of JAX may depend on your CUDA system. Ensure compatibility with your GPU drivers.

### Running Experiments

The workflow is managed by Snakemake. To run the full pipeline (takes approximately 40~60 minutes on an A100 GPU):

```bash
poetry run snakemake --cores 1
```

This will:
- Run simulation experiments for different switchback periods
- Compute ATE

### Plotting Results

To generate the summary plot in Python:

```bash
poetry run python plot.py
```

The output will be saved as `ridesharing.png`.

### Customization

- Edit `Snakefile` to change experiment parameters or add new targets.
- Modify `plot.py` for custom analyses or visualizations.

## Citation

If you use this codebase in your research, please cite:

**Peng, Tianyi, Naimeng Ye, and Andrew Zheng. "Differences-in-Neighbors for Network Interference in Experiments." arXiv preprint arXiv:2503.02271 (2025).**

## License

MIT License (or specify your license here) 