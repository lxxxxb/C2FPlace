# C2FPlace

Official implementation of **"Coarse-to-Fine Macro Placement via Evolutionary Search and Critical Macro Tuning"**.

C2FPlace is a coarse-to-fine hierarchical macro placer. It first builds an initial population of greedy placements (Stage 1: Initialization), then searches macro layouts on a coarse grid with an evolutionary algorithm (Stage 2: Phase 1 global exploration + Phase 2 local refinement), and finally applies critical-macro tuning on the fine grid (Stage 3) to refine HPWL.

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
@misc{liu2026coarsetofinemacroplacementevolutionary,
      title={Coarse-to-Fine Macro Placement via Evolutionary Search and Critical Macro Tuning}, 
      author={Biao Liu and Zhiping Jin and Kaixuan Sun and Zengrui Lu and Qingquan Zhang and Bo Yuan},
      year={2026},
      eprint={2609.34452},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2609.34452}, 
}
```
