# SCDT: Frozen Final Experiments

Reproduction package assembled from the final local paper artifacts on 10 October 2026. The reference figures and Supplementary Material are the `script_0929` submission versions. No scientific results were changed while packaging.

## Scope

This repository supports four distinct operations:

1. Render the final figures from frozen numerical data.
2. Run fresh autonomous SCDT predictions with the frozen fitted models, using the final training-only warm-up protocols.
3. Refit readouts from the included training sequences and frozen recurrent/input/statistical matrices.
4. Recompute rate GT baselines using the included ODE solver versions and fixed initial-condition records.

Readout refitting is **not** a rerun of Bayesian optimization, a resampling of random reservoir matrices, or a regeneration of all training ODE data. Frozen training sequences are included for reproducible fitting. Historical optimization scripts and intermediate candidates are intentionally not executable entry points here. GT selected trajectories and raw-panel measurements used for plotting are included as numerical data; the GT baseline command recomputes rate experiments, not every historical raw-panel search.

## Installation

The assembly/verification environment used Python 3.9.6; package versions are recorded in `requirements-lock.txt` and `provenance/environment.json`. Exact numerical equivalence can depend on BLAS, the ODE solver version and floating-point summation order. No GPU is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run.py test
python run.py verify
```

`requirements.txt` lists the direct dependencies without a version lock. Prefer the lock file when reproducing the frozen results. The runner limits numerical libraries to one thread to preserve the tested summation behavior.

## Commands

Run commands from the repository root. All paths are derived from the package location; no original desktop folder is needed.

### Render final figures (no new simulation or training)

```bash
python run.py figures
python run.py verify --figures outputs/figures
```

Outputs go to `outputs/figures`. Figure 1 is a supplied static diagram, copied without re-simulation. Figures 2--10 are rendered from numeric data; Figure 11 is the auxiliary GT individual-forward-continuation plot. JPG byte equivalence is checked against the included reference files in the tested environment. The submission Figure 3 has external JPEG re-encoding/metadata, so it is checked by dimensions and decoded-pixel error instead; the newly rendered file is not replaced by the reference. On a different rendering stack, layout/font encoding differences may occur even if all numeric data agree.

### Fresh SCDT rate evaluations

The default test/readout policy always selects the submitted model. Running a training command does not change this policy. `--frozen` below is an optional compatibility flag; a refitted readout is used only when explicitly supplied with `--readout`.

```bash
python run.py evaluate --frozen --system food --task rate
python run.py evaluate --frozen --system power --task rate
python run.py evaluate --frozen --system kuramoto --task rate
python run.py verify --predictions outputs/evaluation
```

Default ensembles and horizons are the paper settings. For a quick smoke test only:

```bash
python run.py evaluate --frozen --system food --condition 10 --members 2
python run.py evaluate --frozen --system power --condition 18 --members 2
python run.py evaluate --frozen --system kuramoto --condition 7 --members 2
```

`--condition` can be repeated; indices are zero-based. `--members` evaluates only the first recorded windows and is not a replacement for the paper ensemble. Full sweeps can be computationally expensive, especially the 2,000-node Kuramoto reservoir. Completed condition outputs are saved incrementally.

### Fresh selected predictions and statistical reconstruction

```bash
python run.py evaluate --frozen --system food --task selected
python run.py evaluate --frozen --system power --task selected
python run.py evaluate --frozen --system kuramoto --task selected
python run.py evaluate --frozen --system food --task reconstruction
python run.py evaluate --frozen --system power --task reconstruction
```

Reconstruction uses autonomous outputs only. Collapsed/nonfinite records remain in `measurements_all.csv` but are excluded from `measurements_noncollapsed.csv`. Original plotting/exclusion records remain in the frozen data. Kuramoto pairwise correlation reconstruction is not supported by an r-only readout.

Frozen plot data and fresh statistical measurements are not interchangeable. In the packaging validation, all Power reconstruction standard deviations matched the recorded values. Food preserved all 41 collapse/noncollapse decisions, but fresh standard deviations differed from the historical reconstruction table by up to approximately 0.00198. The verifier reports these Food differences instead of claiming exact equivalence. The paper figures continue to use their original measured tables; new measurements remain separate outputs. See the validation record for the exact comparison scope.

The displayed Food post trajectory uses its recorded 500-step hold and 50-step conditioning ramp; the **Food rate** uses a different, explicitly fixed single-rule protocol with no hold/ramp and no critical-K switch. These two experiment protocols must not be conflated.

### Refit final readouts

```bash
python run.py train --system food
python run.py train --system power
python run.py train --system kuramoto
```

The command saves a newly fitted readout and its relative error against the frozen model in `outputs/training/<system>`. It never replaces frozen model weights. Kuramoto preserves the original batched state collection and eigenvalue-based ridge solution. Power preserves the original chunked sufficient-statistics calculation.

A small relative weight error does not guarantee an identical long autonomous trajectory or rate. Power's refitted weights differ by approximately 3e-9; a near-crossing survival decision can change. Submitted Figure 7 and its Supplementary values must be reproduced with the default submitted model, not an automatically selected refit.

Use a newly fitted readout explicitly in prediction without replacing the frozen checkpoint:

```bash
python run.py evaluate --system food --readout outputs/training/food/readout.npz
```

### Recompute GT rate baselines

```bash
python run.py gt --system food
python run.py gt --system power
python run.py gt --system kuramoto
```

Food and Power use the fixed initial states stored in the baseline member records. Kuramoto starts 20 random-phase realizations at K=0.15 and carries the final phase to the next K through the original 0.15--0.35 continuation; the published Figure 10 uses its 0.19--0.30 subset. Skipping lower-K conditions would change the continuation protocol, so that option is not offered. A one-member check is available via `--members 1`; Food/Power also support `--condition`.

## Final experiment settings

| Setting | Food Chain | Power System | Kuramoto |
|---|---|---|---|
| Training conditions | K: .974, .978, .980, .982, .988 | Q1: 2.989680, 2.989705, 2.989730, 2.989755, 2.989780 | K: .218, .221, .226, .229, .232 |
| RC predicted observable(s) | R, C, P | Four state variables | r(t) only |
| Static channels | std(P), permutation entropy | std(V), lag-1 AC | S_out, S_int |
| Training samples/condition | 2,000 input steps | 60,000 retained samples | 2,301 retained samples |
| Training washout | retain states when t>100 | 100 samples | 80 samples |
| Prediction warm-up | 500 training samples | 100 training samples | 60 training samples |
| Final SCDT rate settings | 22 x 20 windows | 35 x 50 windows | 18 x 20 windows |
| Autonomous rate horizon | 1,000 samples | 2,000 samples | 10,000 samples |
| Decision | all finite; last-500 mean P>.05 | all finite; all 2,000 V>.30 | all finite; last-3,000 fraction r>=.5 is >=.8 |

Detailed reservoir parameters, selected coordinates and time conventions are in `config/final_settings.json`, model parameter records and `docs/SCDT_Supplementary_Material.pdf`.

## Data and code layout

- `System_Food_Chain/train.py`, `System_Power_System/train.py`, `System_Kuramoto/train.py`: system-specific readout fitting.
- Each system's `test.py`: autonomous testing through the shared `common/testing.py` driver.
- `run.py`: optional compatibility runner.
- `common/runtime.py`: independent frozen-model loading, conditioning, warm-up and inference.
- `System_Food_Chain`, `System_Power_System`, `System_Kuramoto`: retained original ODE/reservoir API bodies; unused standalone demonstrations were removed mechanically. Use `run.py` for final settings rather than generic class defaults.
- `common/gt_baselines`: original solver versions used for the GT baselines.
- `common/training.py`: shared fitted-weight saving and verification.
- `tools`: final rendering, GT recomputation and verification commands.
- `models`: final Food (numeric NPZ), Power and Kuramoto models, normalization and training sequences.
- `data`: frozen selected traces, reconstruction measurements, member predictions, GT records and raw statistics.
- `figures/reference`: immutable submission JPGs.
- `docs`: final English Supplementary source/PDF and bibliography.
- `provenance`: source hashes, fixed selection records, assembly environment and verification records.
- `outputs`: generated results, excluded from Git.

Parameter--stat raw panels contain additional GT measurements rather than only training points. Food/Power plotted raw points are individual measurements; the Kuramoto raw panel uses ensemble means. Full candidate and display-selection records are retained. The (b) calibration line aligns the GT parameter positions and the upper axis; SCDT positions remain the supplied statistics. The red line is the displayed SCDT 50% crossing, computed by linear interpolation of adjacent evaluated rate points.

The actual target K/Q1 for a selected trajectory and its axis-calibration label are different fields, not interchangeable estimates of the same input. The supplied second statistic is determined by the final training stat--stat regression, separately from the additional raw-panel regression used for axis alignment.

## Publication checklist

No repository has been created or uploaded by this packaging operation. No license is selected automatically: the authors must approve a license before publishing (see `LICENSE_NOTICE.md`). Do not publish professor correspondence, historical review documents, desktop metadata or private work logs.

The Kuramoto checkpoint is approximately 31 MiB. Use Git rather than the browser uploader for that file, or decide on Git LFS/data hosting. Keep model/data hashes unchanged when moving assets. Run `verify` after copying the package to a new location.

For exact verification scope and any remaining limitations, see `provenance/packaging_validation.json` and the Korean guide in `docs/README_KO.md`.
