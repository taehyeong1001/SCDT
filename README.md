# SCDT

Final code and data for the submitted SCDT experiments. Each system has a `train.py` and a `test.py`; shared functions live in `common`.

```text
System_Food_Chain/     train.py, test.py, Food ODE/reservoir
System_Power_System/   train.py, test.py, Power ODE/reservoir
System_Kuramoto/       train.py, test.py, oscillator/network ODE
common/               shared model loading, warm-up, evaluation and saving
models/               submitted weights and training sequences
data/                 final settings, reference results and raw measurements
tools/                optional figure rendering, GT calculation and verification
```

## Install

Tested with Python 3.9.6. From this directory:

```bash
python -m pip install -r requirements-lock.txt
```

## Reproduce the Submitted Model

```bash
python System_Food_Chain/test.py
python System_Power_System/test.py
python System_Kuramoto/test.py
```

Testing always uses the submitted model by default, even after running `train.py`. It computes fresh autonomous rate predictions using training trajectories only for warm-up. Final ensemble sizes and horizons are retained. The optional `--frozen` flag has the same behavior and remains available for compatibility.

## Refit and Explicitly Test a New Readout

```bash
python System_Food_Chain/train.py
python System_Power_System/train.py
python System_Kuramoto/train.py

# Only --readout opts in to using the newly fitted weights
python System_Power_System/test.py --readout outputs/training/power/readout.npz --output outputs/refit_evaluation
```

Training refits the readout using the included final training sequences and fixed reservoir/input matrices. It does not regenerate the training ODE data, random matrices or Bayesian optimization. New weights are saved to `outputs/training/<system>/readout.npz`; training never replaces the submitted model.

**Power refitting is not guaranteed to reproduce the exact submitted rate.** A relative readout difference of about 3e-9 can affect a near-crossing survival decision in a long autonomous rollout. Use the default submitted model for Figure 7 and Supplementary numbers; use `--readout` only for a separately recorded refitting experiment. The executed model is printed and recorded in the evaluation metadata.

In the tested environment, a fresh 35-condition x 50-member comparison changed only condition 18 from 0.12 (submitted) to 0.26 (refitted); the sigma crossing changed from 0.03273458497180605 to 0.03274668221532637. The corrected default reproduced all submitted Power member trajectories. See `provenance/default_readout_validation.json` for the exact scope and metrics.

```bash
# Submitted model, regardless of any newly trained readout
python System_Food_Chain/test.py --frozen

# Selected inter/pre/post trajectories
python System_Power_System/test.py --task selected

# Supplied versus reproduced statistics (Food/Power only)
python System_Food_Chain/test.py --task reconstruction

# Quick check, not the paper ensemble
python System_Kuramoto/test.py --frozen --condition 7 --members 2

# Readout trained in a nondefault output directory
python System_Food_Chain/test.py --readout /path/to/readout.npz
```

The same test options apply to all systems; Kuramoto does not support pairwise-statistic reconstruction from an r-only output. Conditions are zero-based; `--condition` can be repeated. Outputs are saved to `outputs/evaluation/<system>` unless `--output` is provided.

## Optional Tools

```bash
python tools/render_figures.py
python tools/verify.py
python tools/generate_gt.py --system kuramoto
python run.py test
```

Figure rendering uses frozen numeric data; it does not rerun training or prediction. `run.py` remains as an optional compatibility runner, not a required entry point.

See [the Korean guide](docs/README_KO.md), [detailed protocols](docs/REPRODUCTION.md), [validation](docs/VALIDATION_KO.md) and [Supplementary Material](docs/SCDT_Supplementary_Material.pdf). Food fresh reconstruction std differs from its historical table by up to approximately 0.00198, while all survival decisions agree; reference measurements are not overwritten.

The layout follows the system-specific train/predict and shared-function organization of [Kong et al.'s code repository](https://github.com/lw-kong/Reservoir_with_a_Parameter_Channel_PRR2021). Its MATLAB code and license were not copied; the numerical SCDT implementations and final assets are unchanged.

## License

Original software source code and usage/reproduction documentation are licensed under the [MIT License](LICENSE). Research data, model weights, publication figures and Supplementary Material are outside this grant; see [the scope notice](LICENSE_NOTICE.md). Third-party dependencies retain their own licenses.
