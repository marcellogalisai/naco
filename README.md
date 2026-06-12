# SpongePSO: Research Directions in Sponge Attacks on Early-Exit Neural Networks

This repository contains code and data for training an early-exit CNN on Fashion-MNIST and evaluating multiple black-box sponge-attack methods, including universal adversarial perturbation (UAP) variants.

## What is included

- Source code for:
  - model training
  - attack generation
  - plotting
  - statistical analysis
- The `data/` directory used by the codebase
- A checked-in `results_sponge_workbench/` directory containing a successful sample run with generated artifacts, plots, and statistics

This means the repository already shows concrete system behavior and expected outputs, which helps reproducibility and makes the code easy to inspect before running anything expensive.

## Repository layout

- `sponge_early_exit_workbench.py` — core library: model, attacks, utilities, fixed attack parameters
- `train_sponge_cnn.py` — trains the early-exit CNN and produces model/data artifacts
- `run_sponge_attacks.py` — runs the fixed-parameter attack benchmark
- `plot_sponge_results.py` — generates result visualization
- `analyze_sponge_statistics.py` — generates  statistical analysis
- `data/` — dataset directory used by the scripts
- `results_sponge_workbench/` — sample run outputs already included in the repository
- `latex/` — paper sources

## Environment

Tested with:

- Python `3.10`

Required Python packages:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `Pillow`

### Suggested installation

```bash
cd /path/to/naco
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy pandas matplotlib seaborn tqdm pillow
```

Notes:

- On Apple Silicon, PyTorch may use `mps` automatically when available.
- On other systems, the code falls back to `cuda` or `cpu`.

## Typical workflow

### 1. Train the model and prepare artifacts

```bash
python3 train_sponge_cnn.py
```

This produces:

- trained model weights
- threshold calibration artifacts
- attack candidate splits
- baseline attack-set metrics
- wall-clock budget calibration

### 2. Run attacks

```bash
python3 run_sponge_attacks.py \
  --methods random pso pso_jitter genetic apso clpso universal_ga_weighted universal_pso_jitter_weighted universal_pso_jitter_multiswarm \
  --caps query wall_clock
```

This produces:

- long-form attack CSVs
- attack matrices
- universal perturbation artifacts
- adversarial images
- summary CSVs

### 3. Regenerate paper plots

```bash
python3 plot_sponge_results.py \
  --methods random pso pso_jitter genetic apso clpso universal_ga_weighted universal_pso_jitter_weighted universal_pso_jitter_multiswarm \
  --caps query wall_clock
```

### 4. Regenerate statistical figures

```bash
python3 analyze_sponge_statistics.py
```

## Expected outputs

Main outputs are written under:

- `results_sponge_workbench/`

Important subfolders:

- `results_sponge_workbench/artifacts/`
- `results_sponge_workbench/plots/`
- `results_sponge_workbench/statistics/plots/`

## Sample run

The repository already includes a checked-in `results_sponge_workbench/` directory from a successful run of the pipeline using 5 attack samples. This is included to:

- demonstrate that the code works
- show expected system behavior
- support reproducibility

If you only want to verify that the pipeline runs end-to-end on a small example, use the 5-input smoke test below.

### Command Pipeline

Run the full small-scale pipeline with:

```bash
python3 train_sponge_cnn.py

python3 run_sponge_attacks.py \
  --methods random pso pso_jitter genetic apso clpso universal_ga_weighted universal_pso_jitter_weighted universal_pso_jitter_multiswarm \
  --caps query wall_clock \
  --test-input-cap 5

python3 plot_sponge_results.py \
  --methods random pso pso_jitter genetic apso clpso universal_ga_weighted universal_pso_jitter_weighted universal_pso_jitter_multiswarm \
  --caps query wall_clock

python3 analyze_sponge_statistics.py
```
