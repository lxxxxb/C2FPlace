# C2FPlace

Official implementation of **"Coarse-to-Fine Macro Placement via Evolutionary Search and Critical Macro Tuning"**.

C2FPlace is a coarse-to-fine hierarchical macro placer. It first searches macro layouts on a coarse grid with an evolutionary algorithm (Stage 1 & 2), then applies critical-macro tuning on the fine grid (Stage 3) to refine HPWL.

## Requirements

- Python >= 3.10
- Dependencies: `numpy`, `numba` (see [pyproject.toml](pyproject.toml))

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
source .venv/bin/activate
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy numba
```

## Benchmark Download

The benchmarks (ISPD 2005 suite: `adaptec1`–`adaptec4`, `bigblue1`–`bigblue4`) are **not** included in this repository and must be downloaded separately:

```bash
wget https://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz
tar -xf ispd2005dp.tar.xz
```

After extraction, place the 8 benchmark directories under `benchmark/`, so the layout looks like:

```
benchmark/
├── adaptec1/   # adaptec1.nodes, adaptec1.nets, adaptec1.pl, ...
├── adaptec2/
├── adaptec3/
├── adaptec4/
├── bigblue1/
├── bigblue2/
├── bigblue3/
└── bigblue4/
```

## Quick Start

Run placement on a single benchmark with default settings:

```bash
python run_placement.py --benchmark adaptec1
```

A typical full-configuration run:

```bash
python run_placement.py \
    --benchmark adaptec1 \
    --coarse_grid 512 \
    --pop_size 20 \
    --stage1_iters 5000 \
    --stage2_iters 5000 \
    --stage3_iters 50 \
    --stage3_mode both_2 \
    --stage3_order node_id \
    --ripup_ratio 0.2 \
    --ripup_strategy random \
    --replace_order node_id \
    --ripup_ratio_strategy random_stage \
    --seed 42
```

Results (`.pl` files and HPWL history) are written to `results/<benchmark>/<config_tag>/seed<N>/`.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--benchmark` | `adaptec1` | Benchmark name under `benchmark/` |
| `--coarse_grid` | `256` | Coarse grid size |
| `--pop_size` | `20` | EA population size |
| `--stage1_iters` / `--stage2_iters` | `5000` | Coarse-stage EA iterations |
| `--stage3_iters` | `50` | Critical macro tuning iterations |
| `--stage3_mode` | `both_2` | `coarse`, `fine`, `both_1`, `both_2` |
| `--stage3_order` | `node_id` | Critical macro order: `hpwl`, `random`, `node_id` |
| `--ripup_ratio` | `0.2` | Rip-up ratio for destruction-reconstruction (ignored when `--ripup_ratio_strategy random_stage`) |
| `--ripup_strategy` | `random` | `random` or `bbox` |
| `--replace_order` | `auto` | `auto`, `preserve`, `node_id`, `random`, `area_group` |
| `--ripup_ratio_strategy` | `fixed` | `fixed`, `random_stage`, `decile_random` |
| `--seed` | `42` | Random seed |
| `--output` | auto | Output `.pl` file path |

Run `python run_placement.py --help` for the full list.

**Note on `--ripup_ratio_strategy random_stage`:** the `--ripup_ratio` value is **not used**. Instead, a fresh ratio is sampled every iteration — Stage 1 draws from **U(0.4, 0.7)** and Stage 2 from **U(0.1, 0.4)**. With `fixed`, Stage 1 uses `--ripup_ratio` and Stage 2 uses `max(0.05, ripup_ratio * 0.5)`.

## Batch Experiments

`run_experiments.py` sweeps benchmarks and parameter grids, with optional parallelism:

```bash
# All 8 circuits x 5 seeds (0-4), fully parallel
python run_experiments.py --jobs 40

# Custom subset
python run_experiments.py \
    --benchmarks adaptec1,bigblue1 \
    --seeds 0,1,2 \
    --ripup_ratios 0.2 \
    --coarse_grids 512 \
    --jobs 4
```

Useful flags: `--dry-run` (print commands only), `--continue-on-error`, `--results_dir`. Logs and placements are saved to `results_experiments/<timestamp>/`.

## Project Structure

```
├── c2fplace_ea.py      # Core evolutionary search + critical macro tuning
├── place_db.py         # Benchmark parser (.nodes/.nets/.pl/.scl)
├── evaluator.py        # HPWL / overlap / congestion evaluation (numba-accelerated)
├── run_placement.py    # Single-run entry point
├── run_experiments.py  # Batch experiment runner
└── tests/              # Unit tests
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{c2fplace,
  title   = {Coarse-to-Fine Macro Placement via Evolutionary Search and Critical Macro Tuning},
  note    = {Under review}
}
```
