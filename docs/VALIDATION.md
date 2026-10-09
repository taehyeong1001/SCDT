# Validation and Reproduction Limits

The reference data, models, figures and Supplementary Material retain the submitted scientific results. Documentation and license updates do not change those assets.

## Submitted-Model Evaluation

Fresh autonomous rate evaluation was compared with the stored member trajectories in the recorded Python 3.9.6 environment:

| System | Fresh rate evaluations | Comparison with stored trajectories |
|---|---|---|
| Food Chain | 22 conditions x 20 members | Identical |
| Power System | 35 conditions x 50 members | Identical |
| Kuramoto | 18 conditions x 20 members | Maximum absolute difference about 1.54e-14 |

Rates and interpolated 50% crossings agreed with the reference data. Food/Power selected inter/pre trajectories survived and post trajectories collapsed. Kuramoto selected inter/pre remained asynchronous in all 20 members, while post synchronized in all 20 members.

The corrected default-readout policy passed eight unit tests, including tests that newly trained output files are not selected automatically. See `provenance/packaging_validation.json`, `provenance/system_layout_validation.json` and `provenance/default_readout_validation.json` for the tested scope. This does not guarantee identical results on a different numerical stack.

## Readout Refitting

Food/Kuramoto readout refits had zero relative difference in the tested environment. Power refitting differed by approximately 2.99e-9. A small weight difference need not imply identical long autonomous trajectories or survival decisions.

A fresh Power comparison used 35 conditions x 50 members and 2,000 autonomous samples for each model:

| Metric | Submitted model (default) | Explicitly selected refit |
|---|---|---|
| Condition 18 survival rate | 0.12 | 0.26 |
| SCDT sigma 50% crossing | 0.03273458497180605 | 0.03274668221532637 |
| Calibrated Q1 crossing | 2.989795250 | 2.9897963636363634 |
| Rate MAE (percentage points) | 17.7142857143 | 17.3142857143 |

The other 34 condition rates agreed. All 1,750 default-model trajectories matched the submitted traces. Both SCDT crossings remained below the GT sigma crossing, 0.033059222141686974. Use the default submitted model for paper reproduction; opt into refits with `--readout`. Each evaluation records its selected artifact and SHA256 in `model_selection.json`.

## Statistical Reconstruction and Figures

All 40 Power reconstruction standard deviations and survival decisions matched their reference values. For Food, all 41 survival decisions agreed, but fresh standard deviations differed from the historical table by up to approximately 0.00198. The cause of that difference has not been established. Fresh measurements and the historical figure table remain separate; exact Food reconstruction equivalence is not claimed.

Ten of 11 rendered JPGs were byte-identical to the references in the tested environment. Figure 3 differed only in external JPEG re-encoding: dimensions agreed and the mean decoded RGB difference was approximately 0.423/255. Fonts, plotting libraries and JPEG encoding may affect rendering on other machines.

## GT and Training Scope

Fresh GT checks covered one Food condition/member, one Power condition/member, and one Kuramoto initial-phase realization across all 21 continuation conditions. The full GT ensembles were not recomputed during packaging. The exact recorded Kuramoto K array is reused.

`train.py` refits readouts from packaged sequences and fixed reservoir matrices. It does not rerun Bayesian optimization, generate new random matrices, regenerate all training ODE sequences, or rerun the historical raw-panel point searches. Reference measurements and selection records are retained in `data` and `provenance`.

## Runtime

On 10 October 2026, one recorded rate condition per system was timed with all 20 warm-up windows using Python 3.9.6 on macOS/arm64 and the repository's single-thread settings. Each measurement includes interpreter startup and model loading; no GPU was used.

| System | Timed condition (zero-based) | Autonomous samples/member | Measured one-condition time | Simple full-sweep projection |
|---|---|---|---|---|
| Food Chain | 10 | 1,000 | 9.72 seconds | 22 x 9.72 seconds: about 3.6 minutes |
| Kuramoto | 7 | 10,000 | 24.73 seconds | 18 x 24.73 seconds: about 7.4 minutes |

These are projections, not full-sweep benchmarks. Multiplying startup/model-loading time for every condition overestimates that overhead, while load, conditioning-dependent behavior and hardware differences can change the runtime. The README rounds these to about 4 and 8 minutes. Measurement records are in `provenance/runtime_notes.json`.

Power's approximately 4.5-minute full-rate runtime (35 conditions x 50 members, 2,000 samples/member) was reported by the user, not independently timed in this update. Training, statistical reconstruction, selected trajectories and GT ODE integration have different workloads and are not covered by these estimates.
