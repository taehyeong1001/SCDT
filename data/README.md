# Frozen data

The folder names preserve the final experiment identifiers for traceability. They are not separate alternative paper results.

| Directory | Contents |
|---|---|
| `food_chain` | Five raw training coordinates, selected supplied test coordinates, selected GT/SCDT traces |
| `food_rate_single_rule_20261001` | Final 22 x 20 SCDT member traces, constant-conditioning windows, GT/SCDT rates, raw Food (b) data |
| `voltage_consistent_final` | Final Power selected coordinates/traces and 40 statistical reconstruction measurements |
| `matched_rate_raw_axes_20261001` | Final raw-(b) calibration fits and 35 x 50 Power member traces/tasks |
| `final_figure_notation_20261001` | Final Kuramoto coordinates, selected GT/SCDT figure traces, 18 SCDT and 12 GT rate points and raw (b) means |
| `kuramoto_rate_members` | All 18 x 20 final Kuramoto autonomous rate traces, collected from the original caches |
| `kuramoto_selected` | Selected pre-condition member traces |
| `supplied_vs_reproduced` | Food reconstruction full, noncollapsed, display and excluded tables |
| `gt_baselines` | Original initial-condition/member records and cached Food/Power baseline signals; full Kuramoto continuation summary |
| `raw_candidates` | Additional raw measurements and display-selection candidate audits |

Rates are fractions in the numeric per-condition CSVs. Some combined plot tables use percentages; the rendering scripts preserve the paper's system-specific units. Figures 4/7 use percent and Figure 10 uses fraction.

Display filters do not delete candidates from the archived tables. For Food reconstruction there are 41 measured settings, 40 rows in the display-source table, and 36 visible after the frozen two-axis bounds. Power reconstruction has 40 settings. Raw parameter--stat (b) panels display 30 Food, 40 Power and 40 Kuramoto points; those counts are different from training or reconstruction counts.

`verified_values.json` contains historical comparison fields in addition to final values. Use its `training`, `inter`, `pre`, `post`, `axis_calibration`, final crossing and grid fields for the final experiment. A historical mapped physical-reference field is not the criterion used by the final rate renderer.
