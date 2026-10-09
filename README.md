# SCDT

**How Much Fluctuation Is Dangerous? Predicting Tipping Points via Statistically Conditioned Digital Twins**

Taehyeong Kim and Pilwon Kim, Department of Mathematical Sciences, UNIST.

A statistically conditioned digital twin (SCDT) is a reservoir-computing model conditioned on observable time-series statistics. This repository contains the code, submitted models and reference data for experiments on a food chain, a power system and a Kuramoto oscillator network.

**Paper status:** Submitted manuscript. Publication details and the paper DOI will be added after publication.

## Quick Start

Tested with Python 3.9.6. From the repository root:

```bash
python -m pip install -r requirements-lock.txt

# Small smoke test, not the paper ensemble
python System_Kuramoto/test.py --condition 7 --members 2

# Full rate evaluations with the submitted models
python System_Food_Chain/test.py
python System_Power_System/test.py
python System_Kuramoto/test.py
```

Tests always use the submitted model by default, including after running `train.py`. Warm-up uses training trajectories only, followed by autonomous prediction. Results are saved to `outputs/evaluation/<system>`.

### Expected Runtime

These are approximate wall-clock times for the **full rate sweep**, not training or GT simulation. Hardware, numerical libraries and settings affect runtime; no GPU is required.

| System | Rate sweep | Approximate time |
|---|---|---|
| Food Chain | 22 conditions x 20 members | About 4 minutes (single-condition projection) |
| Power System | 35 conditions x 50 members | About 4.5 minutes (measured full run) |
| Kuramoto | 18 conditions x 20 members | About 8 minutes (single-condition projection) |

Food/Kuramoto estimates come from one full-member condition on macOS/Apple Silicon, not a timed full sweep. Slower machines may require tens of minutes for Kuramoto. See [the timing scope](docs/VALIDATION.md#runtime). Completed rate conditions are saved incrementally.

## Train and Test a New Readout

```bash
python System_Food_Chain/train.py
python System_Power_System/train.py
python System_Kuramoto/train.py

# Explicitly select a newly fitted readout
python System_Power_System/test.py --readout outputs/training/power/readout.npz --output outputs/refit_evaluation
```

Training refits the readout from included training sequences and fixed reservoir matrices; it does not rerun Bayesian optimization or regenerate training ODE data. Weights are saved to `outputs/training/<system>/readout.npz` without replacing the submitted models.

Power refitting can produce small result differences due to chaotic sensitivity. Use the default test commands to reproduce the submitted paper numbers; refitted weights are used only with `--readout`.

## Layout and Other Tasks

```text
System_Food_Chain/     train.py, test.py, Food ODE/reservoir
System_Power_System/   train.py, test.py, Power ODE/reservoir
System_Kuramoto/       train.py, test.py, oscillator/network ODE
common/               shared loading, warm-up, training and evaluation
models/               submitted weights and training sequences
data/                 reference results, settings and raw measurements
tools/                figure rendering, GT calculation and verification
docs/                 reproduction guide, validation and Supplementary
```

```bash
python System_Power_System/test.py --task selected
python System_Food_Chain/test.py --task reconstruction
python tools/render_figures.py
python tools/verify.py
```

Rendering uses saved numerical data, not a new simulation. See [reproduction protocols](docs/REPRODUCTION.md), [validation and limitations](docs/VALIDATION.md), [the Korean guide](docs/README_KO.md) and [Supplementary Material](docs/SCDT_Supplementary_Material.pdf). Additional test options and GT recomputation commands are in the reproduction guide.

## Citation

Please cite the associated manuscript when using this work. A [BibTeX file](CITATION.bib) is included; its publication details will be updated when available.

```bibtex
@unpublished{kim2026scdt,
  author = {Kim, Taehyeong and Kim, Pilwon},
  title = {How Much Fluctuation Is Dangerous? Predicting Tipping Points via Statistically Conditioned Digital Twins},
  year = {2026},
  note = {Submitted manuscript}
}
```

## License and Acknowledgment

Original software source code and usage/reproduction documentation are licensed under the [MIT License](LICENSE). Research data, model weights, publication figures and Supplementary Material are outside this grant; see [the scope notice](LICENSE_NOTICE.md). Third-party dependencies retain their own licenses.

Layout inspired by [Kong et al. (2021)](https://github.com/lw-kong/Reservoir_with_a_Parameter_Channel_PRR2021).
